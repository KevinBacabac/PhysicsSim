def AngleToColour(angle):
    angle %= 360

    saturation = 1
    luminance = 1

    temp1 = luminance * (1 + saturation)
    temp2 = 2 * luminance - temp1
    Hue = angle / 360

    tempR = Hue + 1/3
    tempG = Hue
    tempB = Hue - 1/3

    if tempR < 0:
        tempR += 1
    elif tempG < 0:
        tempG += 1
    elif tempB < 0:
        tempB += 1
    elif tempR > 1:
        tempR -= 1
    elif tempG > 1:
        tempG -= 1
    elif tempG > 1:
        tempG -= 1

    red = ucf(tempR, temp1, temp2)
    green = ucf(tempG, temp1, temp2)
    blue = ucf(tempB, temp1, temp2)

    R = round(red * 255 / 2)
    G = round(green * 255 / 2)
    B = round(blue * 255 / 2)

    return R, G, B


def ucf(tempRGB, temp1, temp2):
    if 6 * tempRGB < 1:
        return temp2 + (temp1 - temp2) * 6 * tempRGB
    elif 2 * tempRGB < 1:
        return temp1
    elif 3 * tempRGB < 2:
        return temp2 + (temp1 - temp2) * (2/3 - tempRGB) * 6
    else:
        return temp2
