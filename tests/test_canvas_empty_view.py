"""Empty-state text keeps its readable size after images and manual zoom."""

import numpy as np
import pytest

from imageguard.ui.widgets.image_canvas import ImageCanvas


@pytest.mark.parametrize("shape,zoom", [((80, 96, 3), 1), ((1600, 2400, 3), 1), ((80, 96, 3), 3)])
def test_empty_view_does_not_inherit_image_zoom(qtbot, shape, zoom):
    canvas = ImageCanvas()
    canvas.resize(800, 600)
    qtbot.addWidget(canvas)
    canvas.show()
    qtbot.waitExposed(canvas)

    def text_rect():
        item = canvas.scene().items()[0]
        return canvas.mapFromScene(item.sceneBoundingRect()).boundingRect()

    initial_size = text_rect().size()
    canvas.set_image(np.zeros(shape, np.uint8))
    canvas.scale(zoom, zoom)
    canvas.clear_image()

    assert canvas._base is None
    assert text_rect().size() == initial_size
    assert canvas.viewport().rect().contains(text_rect())
