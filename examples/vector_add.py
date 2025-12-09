import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def add_kernel(
    x_ptr,  # Pointer to first input vector
    y_ptr,  # Pointer to second input vector
    output_ptr,  # Pointer to output vector
    n_elements,  # Size of the vector
    BLOCK_SIZE: tl.constexpr,  # Number of elements processed by each block
    # NOTE: `constexpr` so it can be used as a shape value
) -> None:
    # Identify position in the grid (which block in the grid)
    # This will start from 0
    pid = tl.program_id(axis=0)
    # We manage the tiles manually
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # offsets for each element in the block
    mask = offsets < n_elements  # mask to avoid out-of-bounds memory access
    # Load x and y from DRAM
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y  # Element-wise addition
    # Store the result back to DRAM
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Preallocate the output tensor
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    # Define the grid
    # grid is either tuple[int] or Callable(metaparams) -> tuple[int]
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # Return a handle to output tensor, but it is still running async at the moment
    # until torch.cuda.synchronize() is called
    return output


def test():
    torch.manual_seed(0)  # does this matter?
    size = 98432  # a non 2^n size to test the mask
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)
    output_torch = x + y
    output_triton = add(x, y)
    print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')


# Benchmark
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # argument names to use as x-axis for the plot
        x_vals=[2**i for i in range(12, 28, 1)],
        x_log=True,
        line_arg='provider',  
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark(size: int, provider: str) -> float:
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    # gbps: effective memory bandwidth = Total Bytes Moved / Total Time
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == '__main__':
    # test()
    benchmark.run(print_data=True, show_plots=True, save_path='results/')
    # show_plots (a pop-up window) won't work in ssh terminal
    pass