from hippocampy import decoding
import unittest
import numpy as np
import bottleneck as bn


class TestDecoding(unittest.TestCase):
    def test_decoded_state_max(self):
        i = np.random.randint(0, 20, (100))
        P_mat = np.zeros((20, 100))
        P_mat[i, range(100)] = 1
        dec = decoding.decoded_state(P_mat, method="max")
        dec_com = decoding.decoded_state(P_mat, method="com")

        assert all(dec == i)
        assert all(dec_com == i)

    def test_confmat(self):
        y_true = [2, 0, 2, 2, 0, 1]
        y_pred = [0, 0, 2, 2, 0, 2]
        expected_out = np.array([[2, 0, 0], [0, 0, 1], [1, 0, 2]])
        cm = decoding.confusion_matrix(y_true, y_pred)
        assert all(expected_out.ravel() == cm.ravel())

    def test_confmat_full(self):
        x_true = [2, 0, 2, 2, 0, 1]
        P = np.array([[2, 0, 0, 1, 2, 0], [0, 0, 0, 0, 1, 1], [0, 1, 0, 6, 1, 1]])
        exp_cm = np.array([[1.0, 0.0, 1.0], [0.5, 1.0, 0.0], [1.0, 1.0, 2.0]])
        cm = decoding.confusion_matrix_full(x_true, P, method="mean")
        assert all(cm.ravel() == exp_cm.ravel())

    def test_frv(self):
        Q = np.array(
            [
                [1, 0, 1, 0, 0.8, 0.0, 0, 0, 0.0],
                [0, 0, 0, 1, 0.9, 0.2, 0, 0, 1.0],
                [0, 1, 0, 0, 1.0, 0.3, 1, 0, 0.0],
                [0, 0, 1, 1, 1.0, 0.0, 0, 1, 0.1],
            ]
        )

        temp = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0],])

        expected = np.array([0, 2, 0, 1, 2, 2, 2, 0, 1])

        a = decoding.frv(Q, temp)
        r_cor = bn.nanargmax(a, axis=0)

        a = decoding.frv(Q, temp, method="cosine")
        r_cos = bn.nanargmax(a, axis=0)

        a = decoding.frv(Q, temp, method="euclidian")
        r_euc = bn.nanargmax(-a, axis=0)

        assert all(r_cor == expected)
        assert all(r_cos == expected)
        assert all(r_euc == expected)
