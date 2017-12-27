
import pytest, os
import numpy as np
from yolo.backend.decoder import YoloDecoder
import yolo

SAMPLE_DIR = os.path.join(yolo.PROJECT_ROOT, "tests", "yolo", "test_sample")


def test_yolo_decoding():
    netout = np.load(os.path.join(SAMPLE_DIR, "decoder_in.npy"))
    yolo_decoder = YoloDecoder()
    boxes, probs = yolo_decoder.run(netout)
    assert np.allclose(boxes, np.array([(0.50070397927, 0.585420268209, 0.680594700387, 0.758197716846)]))
    assert np.allclose(probs, np.array([(0.57606441)]))

if __name__ == '__main__':
    pytest.main([__file__, "-s", "-v"])
