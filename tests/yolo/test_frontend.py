# -*- coding: utf-8 -*-

import pytest
import yolo
import os
from yolo.frontend import get_object_labels


def test_get_object_labels():
    annotation_dir = os.path.join(yolo.PROJECT_ROOT, "tests", "dataset", "svhn", "anns")
    labels = get_object_labels(annotation_dir)
    assert labels == ["1", "2", "3", "9"]
    

if __name__ == '__main__':
    pytest.main([__file__, "-v", "-s"])

