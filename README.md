![Fracted](assets/logo-with-name.svg)

A simple python library for IFS fractals and fractal flames.

It allows you to render fractals as numpy arrays. Then, you can analyze it, display it using matplotlib or save as image with Pillow.

## Installation

Install using `pip`:

```bash
pip install fracted
```

or clone this repo:

```bash
git clone https://github.com/Skrret/fracted
```

and install in editable mode:

```bash
cd fracted
pip install -e .
```

## Usage

```python
from fracted import fractals
```

### Serpinski Triangle


```python
transformations = [
    (lambda point: (point[0] / 2 + 100 * 3 ** 0.5, point[1] / 2)),
    (lambda point: (point[0] / 2, point[1] / 2 + 100)),
    (lambda point: (point[0] / 2, point[1] / 2 - 100)),
]
frac = fractals.IFS(
    transformations,
    min_x=0,
    min_y=-200,
    max_x=400,
    max_y=200,
)
plt.imshow(frac.draw(20, 50_000))
plt.show()
```

![Serpinski triangle](assets/examples/serpinski.png)
    


### Fractal Flames

Fracted can be also used to generate fractal flames.


```python
transformations = [
    (
        lambda point: (
            0.8 * point[0] + 0.5 * numpy.cos(point[1]) - 3,
            0.8 * point[1] + 0.5 * numpy.cos(point[0]) - 3,
        )
    ),
    (
        lambda point: (
            -0.5 * point[1] + 0.5 * abs(point[0]) ** 0.5 + 1,
            0.5 * point[0] + 0.5 * abs(point[0]) ** 0.5 + 3,
        )
    ),
]
frac = fractals.IFS(
    transformations, resolution=50, min_x=-10, min_y=-10, max_x=10, max_y=10
)
plt.imshow(numpy.log10(frac.draw(20, 50000)))
plt.show()
```

![Fractal flame](assets/examples/fractal-flame.png)

More examples are in `examples/`.

## License

[MIT](https://choosealicense.com/licenses/mit/)
