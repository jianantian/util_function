import numpy as np
import math

# 常数, 小于该值的数视为 0
EPSILON = 1e-3

def norm(vec: np.ndarray) -> float:
    """
    向量 vec 的长度
    :param vec:
    :return:
    """
    return math.sqrt(np.sum(vec ** 2))


def normalize(vec: np.ndarray) -> np.ndarray:
    """
    向量 vec 归一化
    :param vec:
    :return:
    """
    return vec / norm(vec)


def normal_vector(vec: np.ndarray) -> np.ndarray:
    """
    返回与 vec 垂直的向量 (vec 逆时针旋转 90 度)
    适用于 2 维向量
    :param vec:
    :return:
    """
    return np.array([-vec[-1], vec[0]])


def get_shift_vec(reference_point, original_point, original_direction):
    """
    计算原三角形的顶点 original_point 相对于三角形中心和 reference_point 的位置
    :param reference_point: 3 维空间中的点, 即三角形绕着转动的顶点
    :param original_point: 待计算的顶点
    :param original_direction: 原三角形中心和 reference_point 连线方向的单位向量
    :return:
    """
    horizontal_shift = np.dot(original_point - reference_point, original_direction)
    _normal_direction = original_point - (reference_point + horizontal_shift * original_direction)

    normal_direction = normalize(_normal_direction)
    vertical_shift = np.dot(original_point - reference_point, normal_direction)

    #     _pt = reference_point + horizontal_shift * original_direction + vertical_shift * normal_direction
    #     print(f'这个距离应该几乎为 0 : {norm(_pt - original_point)}')
    return horizontal_shift, vertical_shift


def _calculate_new_point_scratch(reference_point, original_point, new_direction, horizontal_shift, vertical_shift):
    """
    由之前计算的参数计算转动之后 original_point 对应的新顶点的坐标
    :param reference_point: 3 维空间中的点, 即三角形绕着转动的顶点
    :param original_point: 待计算的顶点, 原来的位置
    :param new_direction: 新三角形中心和 reference_point 连线方向的单位向量
    :param horizontal_shift:
    :param vertical_shift:
    :return:
    """
    normal_direction = np.array([*normal_vector(new_direction[:-1]), 0])
    if np.dot(original_point - reference_point, normal_direction) > 0:
        signature = 1
    else:
        signature = -1
    #     print(f'符号为: {signature}')
    new_point = reference_point + horizontal_shift * new_direction + signature * vertical_shift * normal_direction
    return new_point


def calculate_new_point(reference_point, original_point, original_direction, new_direction):
    """
    计算新三角形的顶点, 将以上两个函数合在一起
    :param reference_point: 3 维空间中的点, 即三角形绕着转动的顶点
    :param original_point: 待计算的顶点, 原来的位置
    :param original_direction: 原三角形中心和 reference_point 连线方向的单位向量
    :param new_direction: 新三角形中心和 reference_point 连线方向的单位向量
    """
    horizontal_shift, vertical_shift =  get_shift_vec(reference_point, original_point, original_direction)
    return _calculate_new_point_scratch(reference_point, original_point, new_direction, horizontal_shift,
                                        vertical_shift)


def flatten_triangle_basic(reference_point, point_b, point_c, center):
    """
    reference_point: 三角形的一个顶点, 三角形绕着转动的点
    point_b, point_c: 三角形另外两个顶点
    center: 三角形的中心
    """
    # z 方向的单位向量
    projection_vector = np.array([0, 0, 1])
    vec = center - reference_point

    # 中心 center 和 reference_point 连成的向量在水平方向的投影
    center_vec = vec - np.dot(projection_vector, vec) * projection_vector
    #     print(f'center_vec: {center_vec}')

    center_length = norm(center_vec)
    if center_length <= EPSILON:
        raise ValueError('三角形竖直')

    scale_para = norm(vec) / center_length
    #     print(f'系数为: {scale_para}')
    center_vec = center_vec * scale_para

    original_direction = normalize(vec)
    #     print(f'orginal_direction: {original_direction}')
    new_direction = normalize(center_vec)
    #     print(f'new_direction: {new_direction}')

    new_point_b = calculate_new_point(reference_point, point_b, original_direction, new_direction)
    new_point_c = calculate_new_point(reference_point, point_c, original_direction, new_direction)
    return (reference_point, new_point_b, new_point_c)


def flatten_triangle(point_a, point_b, point_c):
    """

    :param point_a:
    :param point_b:
    :param point_c:
    :return:
    """
    center = (point_a + point_b + point_c) / 3
    return flatten_triangle_basic(point_a, point_b, point_c, center)


def test_function(point_a, point_b, point_c):
    """
    验证
    :return:
    """
    new_point_a, new_point_b, new_point_c = flatten_triangle(point_a, point_b, point_c)

    assert abs(norm(new_point_b - new_point_a) - norm(point_b - point_a)) < EPSILON
    assert abs(norm(new_point_c - new_point_a) - norm(point_c - point_a)) < EPSILON
    assert abs(norm(new_point_c - new_point_b) - norm(point_c - point_b)) < EPSILON

    assert norm(np.cross(np.cross(new_point_b - new_point_a, new_point_c - new_point_a),
                         np.array([0, 0, 1]))) < EPSILON


if __name__ == '__main__':
    point_a = np.array([3, 1, 0.2])
    point_b = np.array([1.2, 2.5, -1.54])
    point_c = np.array([2.3, 2.6, 2.34])
    test_function(point_a, point_b, point_c)
