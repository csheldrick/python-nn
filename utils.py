def to_rgb(val):
    old = (9.0 - 0.0)
    new = (1.0 - 0.0)
    newv = (((val - 0.0) * new) / old) + 0.0
    return newv


def gen_square(x, y, w, h, size):
    start_x = x
    start_y = y
    coord_x = 0
    coord_y = 0
    while y < h:
        x = start_x
        coord_x = 0
        while x < w:
            yield {'sq':(x, y, size),'coords':(coord_x,coord_y)}
            x += size
            coord_x += 1
        y += size
        coord_y += 1
