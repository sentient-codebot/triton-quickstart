"""In this example, the kernel will implement an online softmax operation,
aiming to break full-row-wise operation into running blocks.
NOTE: In reality, the online softmax is only used inside the fused attention block,
where only the output O is needed. The softmax results are discarded after being
multiplied with the value matrix V. Here we implement a standalone online softmax.
"""

import torch
import triton
import triton.language as tl


DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def online_softmax_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute the softmax of a 1D vector. As a practice of part of Flash Attention. We
    are just computing the normalization factor here."""
    # Initialize running max and sum
    running_m = -float("inf")  # running max
    running_d = 0.0  # running normalized sum of exp

    # Iterate over blocks
    for i in range(0, n_elements, BLOCK_SIZE):
        # Create offsets
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Load block
        x_block = tl.load(
            x_ptr + offsets,
            mask=mask,
            other=-float("inf"),
        )

        # 1. block max
        block_m = tl.max(x_block, axis=0)
        # 2. running max
        new_m = tl.maximum(running_m, block_m)
        # 3. rescale the running sum
        running_d = running_d * tl.exp(running_m - new_m)
        # 4. block exp
        block_d = tl.sum(tl.exp(x_block - new_m), axis=0)
        # 5. update running sum and running max
        running_d += block_d
        running_m = new_m

    # final result
    final_logsumexp = running_m + tl.log(running_d)
    # write back
    tl.store(output_ptr, final_logsumexp)


def test():
    torch.manual_seed(0)
    N = 5000
    BLOCK_SIZE = 1024
    dtype = torch.float32

    x = torch.randn(N, device=DEVICE, dtype=dtype)
    output_triton = torch.empty(1, device=DEVICE, dtype=dtype)

    # torch
    torch_lse = torch.logsumexp(x, dim=0)
    # triton
    grid = (1,)  # in flash attention, the grid is over multiple rows
    online_softmax_kernel[grid](
        x_ptr=x,
        output_ptr=output_triton,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    if torch.allclose(torch_lse, output_triton, rtol=1e-4, atol=1e-4):
        print("✅ SUCCESS: Online Softmax math matches PyTorch!")
    else:
        print("❌ FAILURE: Mismatch detected.")


if __name__ == "__main__":
    test()
