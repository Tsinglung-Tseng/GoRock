from numba import jit, njit, int32, prange
import numpy as np
import math


@njit(parallel=True)
def proj_3d_center(lors, proj, image, pixel_size, center):
    dx, dy, dz = pixel_size
    dz = dz / dx
    nx, ny, nz = image.shape
    nx2, ny2, nz2 = nx / 2, ny / 2, nz / 2

    for i in prange(lors.shape[0]):
        x1, y1, z1 = (lors[i, 0:3] - center) / dx
        x2, y2, z2 = (lors[i, 3:6] - center) / dx

        if (x2 - x1) ** 2 + (y2 - y1) ** 2 > 10:
            if abs(x1 - x2) > abs(y1 - y2):
                ky = (y2 - y1) / (x2 - x1)
                kz = (z2 - z1) / (x2 - x1)
                for ix in range(nx):
                    xx1 = ix - nx2
                    xx2 = xx1 + 1
                    if ky >= 0:
                        yy1 = y1 + ky * (xx1 - x1) + ny2
                        yy2 = y1 + ky * (xx2 - x1) + ny2
                    else:
                        yy1 = y1 + ky * (xx2 - x1) + ny2
                        yy2 = y1 + ky * (xx1 - x1) + ny2
                    cy1 = math.floor(yy1)
                    cy2 = math.floor(yy2)

                    if kz >= 0:
                        zz1 = z1 + kz * (xx1 - x1) + nz2
                        zz2 = z1 + kz * (xx2 - x1) + nz2
                    else:
                        zz1 = z1 + kz * (xx2 - x1) + nz2
                        zz2 = z1 + kz * (xx1 - x1) + nz2
                    cz1 = math.floor(zz1)
                    cz2 = math.floor(zz2)
                    if cy1 == cy2:
                        if cy1 >= 0 and cy1 < ny:
                            iy = cy1
                            if cz1 == cz2:
                                if cz1 >= 0 and cz1 < nz:
                                    iz = cz1
                                    proj[i] += (
                                        ((1 + ky * ky + kz * kz) ** 0.5)
                                        * dx
                                        * image[ix, iy, iz]
                                    )

                            else:
                                if cz1 >= -1 and cz1 < nz:
                                    rz = (cz2 - zz1) / (zz2 - zz1)
                                    if cz1 >= 0:
                                        iz = cz1
                                        proj[i] += (
                                            rz
                                            * ((1 + ky * ky + kz * kz) ** 0.5)
                                            * dx
                                            * image[ix, iy, iz]
                                        )
                                    if cz2 < nz:
                                        iz = cz2
                                        proj[i] += (
                                            (1 - rz)
                                            * ((1 + ky * ky + kz * kz) ** 0.5)
                                            * dx
                                            * image[ix, iy, iz]
                                        )
                    else:
                        if cy1 >= -1 and cy1 < ny:
                            if cz1 == cz2:
                                if cz1 >= 0 and cz1 < nz:
                                    iz = cz1
                                    ry = (cy2 - yy1) / (yy2 - yy1)
                                    if cy1 >= 0:
                                        iy = cy1
                                        proj[i] += (
                                            ry
                                            * ((1 + ky * ky + kz * kz) ** 0.5)
                                            * dx
                                            * image[ix, iy, iz]
                                        )
                                    if cy2 < ny:
                                        iy = cy2
                                        proj[i] += (
                                            (1 - ry)
                                            * ((1 + ky * ky + kz * kz) ** 0.5)
                                            * dx
                                            * image[ix, iy, iz]
                                        )
                            else:
                                if cz1 >= -1 and cz1 < nz:
                                    rz = (cz2 - zz1) / (zz2 - zz1)
                                    ry = (cy2 - yy1) / (yy2 - yy1)
                                    if ry > rz:
                                        if cy1 >= 0 and cz1 >= 0:
                                            iz = cz1
                                            iy = cy1
                                            proj[i] += (
                                                rz
                                                * ((1 + ky * ky + kz * kz) ** 0.5)
                                                * dx
                                                * image[ix, iy, iz]
                                            )
                                        if cy1 >= 0 and cz2 < nz:
                                            iz = cz2
                                            iy = cy1
                                            proj[i] += (
                                                (ry - rz)
                                                * ((1 + ky * ky + kz * kz) ** 0.5)
                                                * dx
                                                * image[ix, iy, iz]
                                            )
                                        if cy2 < ny and cz2 < nz:
                                            iz = cz2
                                            iy = cy2
                                            proj[i] += (
                                                (1 - ry)
                                                * ((1 + ky * ky + kz * kz) ** 0.5)
                                                * dx
                                                * image[ix, iy, iz]
                                            )
                                    else:
                                        if cy1 >= 0 and cz1 >= 0:
                                            iz = cz1
                                            iy = cy1
                                            proj[i] += (
                                                ry
                                                * ((1 + ky * ky + kz * kz) ** 0.5)
                                                * dx
                                                * image[ix, iy, iz]
                                            )
                                        if cy2 < ny and cz1 >= 0:
                                            iz = cz1
                                            iy = cy2
                                            proj[i] += (
                                                (rz - ry)
                                                * ((1 + ky * ky + kz * kz) ** 0.5)
                                                * dx
                                                * image[ix, iy, iz]
                                            )
                                        if cy2 < ny and cz2 < nz:
                                            iz = cz2
                                            iy = cy2
                                            proj[i] += (
                                                (1 - rz)
                                                * ((1 + ky * ky + kz * kz) ** 0.5)
                                                * dx
                                                * image[ix, iy, iz]
                                            )
            else:
                kx = (x2 - x1) / (y2 - y1)
                kz = (z2 - z1) / (y2 - y1)
                for iy in range(ny):
                    yy1 = iy - ny2
                    yy2 = yy1 + 1
                    if kx >= 0:
                        xx1 = x1 + kx * (yy1 - y1) + nx2
                        xx2 = x1 + kx * (yy2 - y1) + nx2
                    else:
                        xx1 = x1 + kx * (yy2 - y1) + nx2
                        xx2 = x1 + kx * (yy1 - y1) + nx2
                    cx1 = math.floor(xx1)
                    cx2 = math.floor(xx2)
                    if kz >= 0:
                        zz1 = z1 + kz * (yy1 - y1) + nz2
                        zz2 = z1 + kz * (yy2 - y1) + nz2
                    else:
                        zz1 = z1 + kz * (yy2 - y1) + nz2
                        zz2 = z1 + kz * (yy1 - y1) + nz2
                    cz1 = math.floor(zz1)
                    cz2 = math.floor(zz2)
                    if cx1 == cx2:
                        if cx1 >= 0 and cx1 < nx:
                            ix = cx1
                            if cz1 == cz2:
                                if cz1 >= 0 and cz1 < nz:
                                    iz = cz1
                                    proj[i] += (
                                        ((1 + kx * kx + kz * kz) ** 0.5)
                                        * dx
                                        * image[ix, iy, iz]
                                    )
                            else:
                                if cz1 >= -1 and cz1 < nz:
                                    rz = (cz2 - zz1) / (zz2 - zz1)
                                    if cz1 >= 0:
                                        iz = cz1
                                        proj[i] += (
                                            rz
                                            * ((1 + kx * kx + kz * kz) ** 0.5)
                                            * dx
                                            * image[ix, iy, iz]
                                        )
                                    if cz2 < nz:
                                        iz = cz2
                                        proj[i] += (
                                            (1 - rz)
                                            * ((1 + kx * kx + kz * kz) ** 0.5)
                                            * dx
                                            * image[ix, iy, iz]
                                        )
                    else:
                        if cx1 >= -1 and cx1 < nx:
                            if cz1 == cz2:
                                if cz1 >= 0 and cz1 < nz:
                                    iz = cz1
                                    rx = (cx2 - xx1) / (xx2 - xx1)
                                    if cx1 >= 0:
                                        ix = cx1
                                        proj[i] += (
                                            rx
                                            * ((1 + kx * kx + kz * kz) ** 0.5)
                                            * dx
                                            * image[ix, iy, iz]
                                        )
                                    if cx2 < nx:
                                        ix = cx2
                                        proj[i] += (
                                            (1 - rx)
                                            * ((1 + kx * kx + kz * kz) ** 0.5)
                                            * dx
                                            * image[ix, iy, iz]
                                        )
                            else:
                                if cz1 >= -1 and cz1 < nz:
                                    rz = (cz2 - zz1) / (zz2 - zz1)
                                    rx = (cx2 - xx1) / (xx2 - xx1)
                                    if rx > rz:
                                        if cx1 >= 0 and cz1 >= 0:
                                            iz = cz1
                                            ix = cx1
                                            proj[i] += (
                                                rz
                                                * ((1 + kx * kx + kz * kz) ** 0.5)
                                                * dx
                                                * image[ix, iy, iz]
                                            )
                                        if cx1 >= 0 and cz2 < nz:
                                            iz = cz2
                                            ix = cx1
                                            proj[i] += (
                                                (rx - rz)
                                                * ((1 + kx * kx + kz * kz) ** 0.5)
                                                * dx
                                                * image[ix, iy, iz]
                                            )
                                        if cx2 < nx and cz2 < nz:
                                            iz = cz2
                                            ix = cx2
                                            proj[i] += (
                                                (1 - rx)
                                                * ((1 + kx * kx + kz * kz) ** 0.5)
                                                * dx
                                                * image[ix, iy, iz]
                                            )
                                    else:
                                        if cx1 >= 0 and cz1 >= 0:
                                            iz = cz1
                                            ix = cx1
                                            proj[i] += (
                                                rx
                                                * ((1 + kx * kx + kz * kz) ** 0.5)
                                                * dx
                                                * image[ix, iy, iz]
                                            )
                                        if cx2 < nx and cz1 >= 0:
                                            iz = cz1
                                            ix = cx2
                                            proj[i] += (
                                                (rz - rx)
                                                * ((1 + kx * kx + kz * kz) ** 0.5)
                                                * dx
                                                * image[ix, iy, iz]
                                            )
                                        if cx2 < nx and cz2 < nz:
                                            iz = cz2
                                            ix = cx2
                                            proj[i] += (
                                                (1 - rz)
                                                * ((1 + kx * kx + kz * kz) ** 0.5)
                                                * dx
                                                * image[ix, iy, iz]
                                            )


@njit
def bproj_3d_center(lors, proj, image, pixel_size, center):
    dx, dy, dz = pixel_size
    nx, ny, nz = image.shape
    nx2, ny2, nz2 = nx / 2, ny / 2, nz / 2

    for i in range(lors.shape[0]):

        x1 = (lors[i, 0] - center[0]) / dx
        y1 = (lors[i, 1] - center[1]) / dx
        z1 = (lors[i, 2] - center[2]) / dx
        x2 = (lors[i, 3] - center[0]) / dx
        y2 = (lors[i, 4] - center[1]) / dx
        z2 = (lors[i, 5] - center[2]) / dx

        if (x2 - x1) ** 2 + (y2 - y1) ** 2 > 10:
            if abs(x1 - x2) > abs(y1 - y2):
                ky = (y2 - y1) / (x2 - x1)
                kz = (z2 - z1) / (x2 - x1)
                for ix in range(nx):
                    xx1 = ix - nx2
                    xx2 = xx1 + 1
                    if ky >= 0:
                        yy1 = y1 + ky * (xx1 - x1) + ny2
                        yy2 = y1 + ky * (xx2 - x1) + ny2
                    else:
                        yy1 = y1 + ky * (xx2 - x1) + ny2
                        yy2 = y1 + ky * (xx1 - x1) + ny2
                    cy1 = math.floor(yy1)
                    cy2 = math.floor(yy2)

                    if kz >= 0:
                        zz1 = z1 + kz * (xx1 - x1) + nz2
                        zz2 = z1 + kz * (xx2 - x1) + nz2
                    else:
                        zz1 = z1 + kz * (xx2 - x1) + nz2
                        zz2 = z1 + kz * (xx1 - x1) + nz2
                    cz1 = math.floor(zz1)
                    cz2 = math.floor(zz2)
                    if cy1 == cy2:
                        if cy1 >= 0 and cy1 < ny:
                            iy = cy1
                            if cz1 == cz2:
                                if cz1 >= 0 and cz1 < nz:
                                    iz = cz1
                                    image[ix, iy, iz] += (
                                        ((1 + ky * ky + kz * kz) ** 0.5) * dx * proj[i]
                                    )

                            else:
                                if cz1 >= -1 and cz1 < nz:
                                    rz = (cz2 - zz1) / (zz2 - zz1)
                                    if cz1 >= 0:
                                        iz = cz1
                                        image[ix, iy, iz] += (
                                            rz
                                            * ((1 + ky * ky + kz * kz) ** 0.5)
                                            * dx
                                            * proj[i]
                                        )
                                    if cz2 < nz:
                                        iz = cz2
                                        image[ix, iy, iz] += (
                                            (1 - rz)
                                            * ((1 + ky * ky + kz * kz) ** 0.5)
                                            * dx
                                            * proj[i]
                                        )
                    else:
                        if cy1 >= -1 and cy1 < ny:
                            if cz1 == cz2:
                                if cz1 >= 0 and cz1 < nz:
                                    iz = cz1
                                    ry = (cy2 - yy1) / (yy2 - yy1)
                                    if cy1 >= 0:
                                        iy = cy1
                                        image[ix, iy, iz] += (
                                            ry
                                            * ((1 + ky * ky + kz * kz) ** 0.5)
                                            * dx
                                            * proj[i]
                                        )
                                    if cy2 < ny:
                                        iy = cy2
                                        image[ix, iy, iz] += (
                                            (1 - ry)
                                            * ((1 + ky * ky + kz * kz) ** 0.5)
                                            * dx
                                            * proj[i]
                                        )
                            else:
                                if cz1 >= -1 and cz1 < nz:
                                    rz = (cz2 - zz1) / (zz2 - zz1)
                                    ry = (cy2 - yy1) / (yy2 - yy1)
                                    if ry > rz:
                                        if cy1 >= 0 and cz1 >= 0:
                                            iz = cz1
                                            iy = cy1
                                            image[ix, iy, iz] += (
                                                rz
                                                * ((1 + ky * ky + kz * kz) ** 0.5)
                                                * dx
                                                * proj[i]
                                            )
                                        if cy1 >= 0 and cz2 < nz:
                                            iz = cz2
                                            iy = cy1
                                            image[ix, iy, iz] += (
                                                (ry - rz)
                                                * ((1 + ky * ky + kz * kz) ** 0.5)
                                                * dx
                                                * proj[i]
                                            )
                                        if cy2 < ny and cz2 < nz:
                                            iz = cz2
                                            iy = cy2
                                            image[ix, iy, iz] += (
                                                (1 - ry)
                                                * ((1 + ky * ky + kz * kz) ** 0.5)
                                                * dx
                                                * proj[i]
                                            )
                                    else:
                                        if cy1 >= 0 and cz1 >= 0:
                                            iz = cz1
                                            iy = cy1
                                            image[ix, iy, iz] += (
                                                ry
                                                * ((1 + ky * ky + kz * kz) ** 0.5)
                                                * dx
                                                * proj[i]
                                            )
                                        if cy2 < ny and cz1 >= 0:
                                            iz = cz1
                                            iy = cy2
                                            image[ix, iy, iz] += (
                                                (rz - ry)
                                                * ((1 + ky * ky + kz * kz) ** 0.5)
                                                * dx
                                                * proj[i]
                                            )
                                        if cy2 < ny and cz2 < nz:
                                            iz = cz2
                                            iy = cy2
                                            image[ix, iy, iz] += (
                                                (1 - rz)
                                                * ((1 + ky * ky + kz * kz) ** 0.5)
                                                * dx
                                                * proj[i]
                                            )
            else:
                kx = (x2 - x1) / (y2 - y1)
                kz = (z2 - z1) / (y2 - y1)
                for iy in range(ny):
                    yy1 = iy - ny2
                    yy2 = yy1 + 1
                    if kx >= 0:
                        xx1 = x1 + kx * (yy1 - y1) + nx2
                        xx2 = x1 + kx * (yy2 - y1) + nx2
                    else:
                        xx1 = x1 + kx * (yy2 - y1) + nx2
                        xx2 = x1 + kx * (yy1 - y1) + nx2
                    cx1 = math.floor(xx1)
                    cx2 = math.floor(xx2)
                    if kz >= 0:
                        zz1 = z1 + kz * (yy1 - y1) + nz2
                        zz2 = z1 + kz * (yy2 - y1) + nz2
                    else:
                        zz1 = z1 + kz * (yy2 - y1) + nz2
                        zz2 = z1 + kz * (yy1 - y1) + nz2
                    cz1 = math.floor(zz1)
                    cz2 = math.floor(zz2)
                    if cx1 == cx2:
                        if cx1 >= 0 and cx1 < nx:
                            ix = cx1
                            if cz1 == cz2:
                                if cz1 >= 0 and cz1 < nz:
                                    iz = cz1
                                    image[ix, iy, iz] += (
                                        ((1 + kx * kx + kz * kz) ** 0.5) * dx * proj[i]
                                    )
                            else:
                                if cz1 >= -1 and cz1 < nz:
                                    rz = (cz2 - zz1) / (zz2 - zz1)
                                    if cz1 >= 0:
                                        iz = cz1
                                        image[ix, iy, iz] += (
                                            rz
                                            * ((1 + kx * kx + kz * kz) ** 0.5)
                                            * dx
                                            * proj[i]
                                        )
                                    if cz2 < nz:
                                        iz = cz2
                                        image[ix, iy, iz] += (
                                            (1 - rz)
                                            * ((1 + kx * kx + kz * kz) ** 0.5)
                                            * dx
                                            * proj[i]
                                        )
                    else:
                        if cx1 >= -1 and cx1 < nx:
                            if cz1 == cz2:
                                if cz1 >= 0 and cz1 < nz:
                                    iz = cz1
                                    rx = (cx2 - xx1) / (xx2 - xx1)
                                    if cx1 >= 0:
                                        ix = cx1
                                        image[ix, iy, iz] += (
                                            rx
                                            * ((1 + kx * kx + kz * kz) ** 0.5)
                                            * dx
                                            * proj[i]
                                        )
                                    if cx2 < nx:
                                        ix = cx2
                                        image[ix, iy, iz] += (
                                            (1 - rx)
                                            * ((1 + kx * kx + kz * kz) ** 0.5)
                                            * dx
                                            * proj[i]
                                        )
                            else:
                                if cz1 >= -1 and cz1 < nz:
                                    rz = (cz2 - zz1) / (zz2 - zz1)
                                    rx = (cx2 - xx1) / (xx2 - xx1)
                                    if rx > rz:
                                        if cx1 >= 0 and cz1 >= 0:
                                            iz = cz1
                                            ix = cx1
                                            image[ix, iy, iz] += (
                                                rz
                                                * ((1 + kx * kx + kz * kz) ** 0.5)
                                                * dx
                                                * proj[i]
                                            )
                                        if cx1 >= 0 and cz2 < nz:
                                            iz = cz2
                                            ix = cx1
                                            image[ix, iy, iz] += (
                                                (rx - rz)
                                                * ((1 + kx * kx + kz * kz) ** 0.5)
                                                * dx
                                                * proj[i]
                                            )
                                        if cx2 < nx and cz2 < nz:
                                            iz = cz2
                                            ix = cx2
                                            image[ix, iy, iz] += (
                                                (1 - rx)
                                                * ((1 + kx * kx + kz * kz) ** 0.5)
                                                * dx
                                                * proj[i]
                                            )
                                    else:
                                        if cx1 >= 0 and cz1 >= 0:
                                            iz = cz1
                                            ix = cx1
                                            image[ix, iy, iz] += (
                                                rx
                                                * ((1 + kx * kx + kz * kz) ** 0.5)
                                                * dx
                                                * proj[i]
                                            )
                                        if cx2 < nx and cz1 >= 0:
                                            iz = cz1
                                            ix = cx2
                                            image[ix, iy, iz] += (
                                                (rz - rx)
                                                * ((1 + kx * kx + kz * kz) ** 0.5)
                                                * dx
                                                * proj[i]
                                            )
                                        if cx2 < nx and cz2 < nz:
                                            iz = cz2
                                            ix = cx2
                                            image[ix, iy, iz] += (
                                                (1 - rz)
                                                * ((1 + kx * kx + kz * kz) ** 0.5)
                                                * dx
                                                * proj[i]
                                            )


def pb_image(
    lors,
    center=np.array([0, 0, 0]),
    shape=np.array([100, 40, 100]) * 2,
    pixel_size=np.array([0.2, 0.2, 0.2]),
):
    # center = np.array([0,0,0])
    # shape = np.array([100,40,100])*2
    # pixel_size = np.array([0.2,0.2,0.2])
    image_bp = np.zeros(shape)
    project_value = np.ones(lors.shape[0])
    bproj_3d_center(lors, project_value, image_bp, pixel_size, center)
    return image_bp
