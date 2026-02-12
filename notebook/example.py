import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    from skimage.io import imread
    from skimage.util import img_as_float
    from skimage.transform import resize
    from sklearn.cluster import KMeans
    from skimage.feature import match_template
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import distance_transform_edt
    from PIL import Image
    import io
    from urllib.request import urlopen

    return (
        Image,
        distance_transform_edt,
        img_as_float,
        imread,
        io,
        match_template,
        mo,
        np,
        plt,
        resize,
    )


@app.cell(hide_code=True)
def _(mo):
    palette_file = mo.ui.file( label="Upload an image to process")
    target_file = mo.notebook_location() / "public" / "mylove_100x100.jpg"
    mask_target = mo.notebook_location() / "public" / "mask_mylove_100x100.tiff"
    run_project = mo.ui.run_button(label="Place Yourself!")
    mo.vstack(
        [
            mo.md(f"""
            # You're 1 in 4 Million
            In reality you're way more special than that... But it makes sense for the project, so please bear with me.
            Upload an image below! Afterwards when you're ready, press the *button* to begin! Enjoy my love! I hope it's fun!

        """),
            palette_file,
            run_project,
        ]
    )
    return mask_target, palette_file, run_project, target_file


@app.cell(hide_code=True)
def _(fixed, mo, plt):
    plt.figure(figsize=(8,8))
    plt.imshow(fixed)
    plt.xticks([])
    plt.yticks([])
    hint1 = mo.ui.run_button(label="Hint 1 :)")
    hint2 = mo.ui.run_button(label="Hint 2 :|")
    hint3 = mo.ui.run_button(label="Hint 3 :(")
    giveup = mo.ui.run_button(label="Get Solution... :((((((((")
    mo.vstack([
        mo.md(f"""
            # Nice You Got this far!
            This should be the image you picked!  Cool trick right?  Lol just kidding, look closer at the picture.
            There should be an interesting addition!  (Really hoping it wasn't to obvious..).  If you're still unsure whats going on check out the hints below! 
        """),
        mo.hstack([plt.gca(),
        mo.vstack([hint1,hint2,hint3,giveup],align='start')])
    ])
    return giveup, hint1, hint2, hint3


@app.cell(hide_code=True)
def _(hint1, mo, moving, plt):
    mo.stop(not hint1.value)
    plt.imshow(moving)
    mo.vstack([
        mo.md(f"""
            # Hint 1
            Okokok, I hope it's not gonna be too tough, but suprise! You're trying to find *YOURSELF!*. Well a very small and
            pixelated version of yourself.  I realize it's quite pixelated and I'm sorry about that but I had to do some heavy resizing to make this run quickly. So here you are!  A pixelated cutie <3
        """),
        plt.gca()

    ])
    return


@app.cell(hide_code=True)
def _(hint2, mo, plt, result):
    mo.stop(not hint2.value)
    plt.imshow(result)
    mo.vstack([
        mo.md(f"""
            # Hint 2
            Okokok, my bad this might be a tougher picture... But 1 thing I did to make it harder was colour match your picture with the input image.  The result will be below!  (If the image has a lot of dark background things go sideways sorry!)
        """),
        plt.gca()

    ])
    return


@app.cell(hide_code=True)
def _(hint3, mo, plt, score):
    mo.stop(not hint3.value)
    plt.imshow(score, cmap="gray")
    mo.vstack([
        mo.md(f"""
            # Hint 3
            So this is obviously a hard picture!  Well Shit!  Hmmm the last piece of help I can give you is where to start looking. I've put that below, bright pixels mean a good place to look!  To give you more insight, I search for the patch of the image you sent that's most correlated with the picture of you! Each pixel in this image is the correlation when the top left of the picture of you is there! Happy Hunting!
        """),
        plt.gca()

    ])
    return


@app.cell(hide_code=True)
def _(fixed, giveup, mo, palette, resize):
    mo.stop(not giveup.value)
    mo.vstack([
        mo.md(f"""
            # Solution
            Damn!  I'm sorry it's so tough, let's see if we can at least lyk where it is!  Below will be a comparison image.  You'll see a slider in the middle you can move left to right.  All of the image to the left of the slider will be the original and all the image to the right of the slider will be from the image with you added!  Hope you can find the solution!
        """),
        mo.image_compare(resize(palette,fixed.shape),fixed)
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(f"""
    ## Whoaaaaa Why you looking down here??
    only boring code below here!
    ... Also the answer to the 1 in 4 million thing ...
    """)
    return


@app.cell(hide_code=True)
def _(
    Image,
    img_as_float,
    imread,
    io,
    mask_target,
    mo,
    np,
    palette_file,
    target_file,
):
    mo.stop(len(palette_file.value) == 0)
    bytes_array = palette_file.value[0].contents
    im = Image.open(io.BytesIO(bytes_array))
    palette = img_as_float(np.array(im))

    image_bytes = target_file.read_bytes()
    mask_bytes = mask_target.read_bytes()

    template = img_as_float(imread(io.BytesIO(image_bytes)))
    mask = img_as_float(imread(io.BytesIO(mask_bytes)))[:,:,:3] > 0.5
    return mask, palette, template


@app.cell(hide_code=True)
def _(
    distance_transform_edt,
    mask,
    match_template,
    mo,
    np,
    palette,
    palette_file,
    pixel_sorting,
    resize,
    run_project,
    template,
):
    mo.stop(len(palette_file.value) == 0 or not run_project.value)
    fixed = resize(palette, (2048,2048), anti_aliasing=True)
    moving = resize(template, (75,75), anti_aliasing=True)
    # 1 in 4 million since 2048*2048 = 4 194 304, should I divide by 75^2? yes but also 1 in 700 sounds way less cool!

    # Find Patch top-left with Highest Correlation
    score = match_template(fixed.copy(), moving.copy())
    tl = np.unravel_index(np.argmax(score), score.shape)

    # Extract patch and palette transfer to target
    roi = fixed[tl[0]:tl[0] + moving.shape[0],tl[1]:tl[1]+moving.shape[1],:]
    result,_ = pixel_sorting(roi, moving)

    # Blending Alpha Matrix --> using mask
    resize_mask = resize(mask[:,:,0], moving.shape[:2]) == 0
    dist = np.sqrt(distance_transform_edt(resize_mask))
    blend_mask = np.stack([dist,dist,dist],axis = 2) / dist.max()

    fixed[tl[0]:tl[0] + moving.shape[0],tl[1]:tl[1]+moving.shape[1],:] = (blend_mask)*roi + (1 - blend_mask) * result
    return fixed, moving, result, score


@app.cell(hide_code=True)
def _(gray2rgb, np, resize):
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    from skimage.filters import sobel
    from typing import Callable


    def flatten(img: np.ndarray):
        """
        Wrapper around array flatten so I can
        overload it

        :param img: Description
        :type img: np.ndarray
        """
        return img.flatten()


    def edge_magnitude(img):
        if img.ndim == 3:
            return flatten(np.sum(np.abs(sobel(img)), axis=2))
        return flatten(np.abs(sobel(img)))


    def luminance(img: np.ndarray):
        if img.ndim == 2:
            return img
        return np.dot(img[..., :3], [0.299, 0.587, 0.114])


    def pixel_sorting(
        img: np.ndarray,
        target: np.ndarray,
        score_func: Callable = flatten,
        colour_func: Callable = luminance,
    ) -> tuple[np.ndarray, tuple[np.ndarray]]:
        """
        Docstring for pixel_sorting
        This function calculated the way to make 1 image look like another using
        pallette sorting

        :param img: Description
        :type img: np.ndarray
        :param target: Description
        :type target: np.ndarray
        """

        if target.ndim == 3 and img.ndim == 2:
            img = gray2rgb(img)

        input_img = np.array(resize(img, target.shape, anti_aliasing=True))

        # Handle Colour
        if input_img.ndim > 2:
            palette = colour_func(input_img)
        else:
            palette = input_img

        if target.ndim > 2:
            template = colour_func(target)
        else:
            template = target

        input_vec = palette.flatten()

        # Compute features to sort on
        input_scores = score_func(palette)
        target_scores = score_func(template)

        input_idx = np.argsort(input_scores)
        target_idx = np.argsort(target_scores)

        # reshape result
        if input_img.ndim <= 2:
            result = np.zeros_like(input_vec)
            result[target_idx] = input_vec[input_idx]
            return result.reshape(input_img.shape), input_idx, target_idx
        input_y, input_x = np.unravel_index(input_idx, palette.shape)
        target_y, target_x = np.unravel_index(target_idx, template.shape)
        result = np.zeros_like(target)
        result[target_y, target_x, :] = input_img[input_y, input_x, :]
        return (
            result,
            (input_y, input_x, target_y, target_x),
        )

    return (pixel_sorting,)


if __name__ == "__main__":
    app.run()
