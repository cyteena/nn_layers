import marimo

__generated_with = "0.16.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    from einops.layers.torch import Rearrange
    from einops import rearrange
    return Rearrange, rearrange, torch


@app.cell
def _(torch):
    dim = 128
    dim_inner = 1024

    img_size = 128
    patch_size = 4

    img = torch.randn((3, img_size, img_size))
    return img, patch_size


@app.cell
def _(img, patch_size, rearrange):
    patches = rearrange(img, "c (h p1) (w p2) -> h w (c p1 p2)", p1=patch_size, p2=patch_size)
    return (patches,)


@app.cell
def _(patches):
    patches.shape
    return


@app.cell
def _(Rearrange, img, patch_size):
    Rearrange("c (h p1) (w p2) -> h w (c p1 p2)", p1=patch_size, p2=patch_size)(img)
    return


app._unparsable_cell(
    r"""
    patches[0][0] == 
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
