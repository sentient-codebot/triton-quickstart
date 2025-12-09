import torch
import triton
import triton.language as tl
from triton.runtime import driver


DEVICE = triton.runtime.driver.active.get_active_torch_device()


def is_hip():
    return False  # I have no AMD gpu


def is_cdna():
    return False  # I have no AMD gpu


def naive_softmax(x):
    """Compute row-wise softmax of X using native pytorch

    We use the safe-softmax by substracting the max for numerical stability

    `x`: (M, N) tensor
    """
    # read MN elements, write M elements
    x_max = x.max(dim=1)[0]
    # read MN + M elements, write MN elements
    z = x - x_max[:, None]
    # read MN elements, write MN elements
    numerator = torch.exp(z)
    # read MN elements, write M elements
    denomintor = numerator.sum(dim=1)
    # read MN + M elements, write MN elements
    ret = numerator / denomintor[:, None]
    # in total: read 5MN + 2M elements, write 3MN + 2M elements
    return ret


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
) -> None:
    """
    the stride represents how much we need to increase the point to advance one row"""
    # starting row of the program ~= tile
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets  # pointers to each element in the row
        # BLOCK_SIZE maybe > than n_cols. But design wise this should be avoided.
        # for this kernel, its row-wise global so BLOCK_SIZE has to be >= n_cols.
        # ideally, k = 1.
        mask = col_offsets < n_cols
        # read BLOCK_SIZE elements
        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        # Subtract max
        # all elements are already in registers, so no extra memory access
        row_minus_max = row - tl.max(row, axis=0)
        # Exponentiate
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back to DRAM (VRAM)
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


# Helper function to enqueue the kernel and its meta arguments
properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}


def softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    num_warps = 8

    num_stages = 2

    # Allocate output
    y = torch.empty_like(x)

    # pre-compile the kernel
    kernel = softmax_kernel.warmup(
        y,
        x,
        x.stride(0),  # let torch calculates in case of non row-major tensor
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,
        grid=(1,),  # one block
    )
    kernel._init_handles()
    n_regs = kernel.n_regs  # "My compiled code needs 36 registers per thread."
    size_smem = (
        kernel.metadata.shared
    )  # "My compiled code needs 4128 bytes of Shared Memory per block."
    if is_hip():
        raise NotImplementedError
    # occupancy: How many of these blocks can we physically jam onto one SM?
    occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    occupancy = min(occupancy, SIZE_SMEM // size_smem)
    num_programs = NUM_SM * occupancy

    num_programs = min(num_programs, n_rows)

    # Create a number of persistent programs
    kernel[(num_programs, 1, 1)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE,
        num_stages,
    )  # NOTE: can only pass positions args!
    return y


# unit test
def test():
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device=DEVICE)
    y_triton = softmax(x)
    y_torch = torch.softmax(x, axis=1)
    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)


# benchmark
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],  # argument names to use as an x-axis for the plot
        x_vals=[
            128 * i for i in range(2, 100)
        ],  # different possible values for `x_name`
        line_arg="provider",  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            "triton",
            "torch",
            "naive_softmax",
        ],  # possible values for `line_arg``
        line_names=["Triton", "Torch", "Naive Softmax"],  # label name for the lines
        styles=[("blue", "-"), ("green", "-"), ("red", "-")],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={"M": 4096},  # values for function arguments not in `x_names` and `y_name`
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == "torch":
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == "triton":
        ms = triton.testing.do_bench(lambda: softmax(x))
    if provider == "naive_softmax":
        ms = triton.testing.do_bench(lambda: naive_softmax(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


benchmark.run(show_plots=True, print_data=True)


if __name__ == "__main__":
    # test()
    benchmark.run(show_plots=True, print_data=True, save_path="results/")