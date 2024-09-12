import numpy as np
import pytest
from utils import *

def test_dot_product():
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    
    result = dot_product(vector1, vector2)
    
    assert result == 32, f"Expected 32, but got {result}"
    
def test_cosine_similarity():
    vector1 = np.array([1, 0])
    vector2 = np.array([0, 1])
    
    result = cosine_similarity(vector1, vector2)
    
    # Cosine similarity of orthogonal vectors (90 degrees) should be 0
    expected_result = 0.0
    
    assert np.isclose(result, expected_result), f"Expected {expected_result}, but got {result}"

def test_nearest_neighbor():
    point = np.array([1, 2])
    points = np.array([[1, 1], [2, 2], [3, 3]])
    
    result = nearest_neighbor(point, points)
    
    # Nearest neighbor to [1, 2] should be [1, 1] (index 0)
    expected_index = 0
    
    assert result == expected_index, f"Expected index {expected_index}, but got {result}"
