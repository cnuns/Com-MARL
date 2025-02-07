from PIL import Image, ImageDraw


def draw_grid(rows, cols, cell_size=50, fill='black', line_color='black'):
    height = rows * cell_size
    width = cols * cell_size
    image = Image.new(mode='RGB', size=(width, height), color=fill)

    # Draw some lines
    draw = ImageDraw.Draw(image)
    y_start = 0
    y_end = image.height
    step_size = cell_size

    for x in range(0, image.width, step_size):
        line = ((x, y_start), (x, y_end))
        draw.line(line, fill=line_color)

    x = image.width - 1
    line = ((x, y_start), (x, y_end))
    draw.line(line, fill=line_color)

    x_start = 0
    x_end = image.width

    for y in range(0, image.height, step_size):
        line = ((x_start, y), (x_end, y))
        draw.line(line, fill=line_color)

    y = image.height - 1
    line = ((x_start, y), (x_end, y))
    draw.line(line, fill=line_color)

    del draw

    return image


def fill_cell(image, pos, cell_size=None, fill='black', margin=0):
    assert cell_size is not None and 0 <= margin <= 1

    col, row = pos
    row, col = row * cell_size, col * cell_size
    margin *= cell_size
    x, y, x_dash, y_dash = row + margin, col + margin, row + cell_size - margin, col + cell_size - margin
    ImageDraw.Draw(image).rectangle([(x, y), (x_dash, y_dash)], fill=fill)


def write_cell_text(image, text, pos, cell_size=None, fill='black', margin=0):
    assert cell_size is not None and 0 <= margin <= 1

    col, row = pos
    row, col = row * cell_size, col * cell_size
    margin *= cell_size
    x, y = row + margin, col + margin
    ImageDraw.Draw(image).text((x, y), text=text, fill=fill)


def draw_cell_outline(image, pos, cell_size=50, fill='black'):
    col, row = pos
    row, col = row * cell_size, col * cell_size
    ImageDraw.Draw(image).rectangle([(row, col), (row + cell_size, col + cell_size)], outline=fill, width=3)


def draw_sensing_outline(image, pos, Rsen, width=1, cell_size=50, fill='black'):
    col, row = pos
    offset = 1
    x1, y1 = (row-Rsen) * cell_size + offset, (col-Rsen) * cell_size + offset
    x2, y2 = (row+Rsen+offset) * cell_size - offset, (col+Rsen+offset) * cell_size - offset
    ImageDraw.Draw(image).rectangle([(x1, y1), (x2, y2)], outline=fill, width=width)

def draw_circle(image, pos, cell_size=50, fill='black', radius=0.3, outline=None):
    col, row = pos
    row, col = row * cell_size, col * cell_size
    gap = cell_size * radius
    x, y = row + gap, col + gap
    x_dash, y_dash = row + cell_size - gap, col + cell_size - gap
    ImageDraw.Draw(image).ellipse([(x, y), (x_dash, y_dash)], outline=fill, fill=fill)

def draw_circle_border(image, pos, cell_size=50, fill='black', radius=0.3, outline=None):
    col, row = pos
    row, col = row * cell_size, col * cell_size
    gap = cell_size * radius
    x, y = int(row + gap), int(col + gap)
    x_dash, y_dash = int(row + cell_size - gap), int(col + cell_size - gap)
    ImageDraw.Draw(image).ellipse([(x, y), (x_dash, y_dash)], outline=outline)

def draw_border(image, border_width=1, fill='black'):
    width, height = image.size
    new_im = Image.new("RGB", size=(width + 2 * border_width, height + 2 * border_width), color=fill)
    new_im.paste(image, (border_width, border_width))
    return new_im


def draw_score_board(image, score, board_height=30):
    im_width, im_height = image.size
    new_im = Image.new("RGB", size=(im_width, im_height + board_height), color='#e1e4e8')
    new_im.paste(image, (0, board_height))

    _text = ', '.join([str(round(x, 2)) for x in score])
    ImageDraw.Draw(new_im).text((10, board_height // 3), text=_text, fill='black')
    
    #ImageDraw.Draw(new_im).text((10, board_height // 3), text=score, fill='black')
    return new_im

def draw_triangle(image, pos, cell_size=50, fill='black', triangle_size=0.25):
    col, row = pos
    row, col = row * cell_size, col * cell_size
    gap = cell_size * triangle_size
    
    # Calculate the vertex coordinates of a triangle.
    x1, y1 = row + gap, col + cell_size - gap
    x2, y2 = row + cell_size // 2, col + gap
    x3, y3 = row + cell_size - gap, col + cell_size - gap
    
    # Draw the triangle.
    ImageDraw.Draw(image).polygon([(x1, y1), (x2, y2), (x3, y3)], outline=fill, fill=fill)