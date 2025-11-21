# =================================  TESTY  ===================================
# Testy do tego pliku zostały podzielone na dwie kategorie:
#
#  1. `..._invalid_input`:
#     - Sprawdzające poprawną obsługę nieprawidłowych danych wejściowych.
#
#  2. `..._correct_solution`:
#     - Weryfikujące poprawność wyników dla prawidłowych danych wejściowych.
# =============================================================================
import numpy as np
import numpy.polynomial.polynomial as nppoly


def roots_20(coef: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if not isinstance(coef, np.ndarray) or coef.ndim != 1:
        return None
    
    disturbance = np.random.randn(coef.size) * 1e-10
    coef_disturbed = coef + disturbance
    
    roots = nppoly.polyroots(coef_disturbed)

    return coef_disturbed, roots
    """Funkcja wyznaczająca miejsca zerowe wielomianu funkcją
    `nppoly.polyroots()`, najpierw lekko zaburzając wejściowe współczynniki 
    wielomianu (N(0,1) * 1e-10).

    Args:
        coef (np.ndarray): Wektor współczynników wielomianu (n,).

    Returns:
        (tuple[np.ndarray, np. ndarray]):
            - Zaburzony wektor współczynników (n,),
            - Wektor miejsc zerowych (m,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """


def frob_a(coef: np.ndarray) -> np.ndarray | None:
    """Funkcja służąca do wyznaczenia macierzy Frobeniusa na podstawie
    współczynników jej wielomianu charakterystycznego:
    w(x) = a_n*x^n + a_{n-1}*x^{n-1} + ... + a_2*x^2 + a_1*x + a_0

    Testy wymagają poniższej definicji macierzy Frobeniusa (implementacja dla 
    innych postaci nie jest zabroniona):
    F = [[       0,        1,        0,   ...,            0],
         [       0,        0,        1,   ...,            0],
         [       0,        0,        0,   ...,            0],
         [     ...,      ...,      ...,   ...,          ...],
         [-a_0/a_n, -a_1/a_n, -a_2/a_n,   ..., -a_{n-1}/a_n]]

    Args:
        coef (np.narray): Wektor współczynników wielomianu (n,).

    Returns:
        (np.ndarray): Macierz Frobeniusa o rozmiarze (n,n).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not isinstance(coef, np.ndarray) or coef.ndim != 1:
        return None
    if coef.size < 2 or coef[0] == 0:
        return None

    n = coef.size - 1

    if coef.size == 2:
        return np.array([[-coef[1] / coef[0]]])

    F = np.zeros((n, n), dtype=float)

    for i in range(n - 1):
        F[i, i + 1] = 1

    F[-1, :] = -coef[1:] / coef[0]

    return F



def is_nonsingular(A: np.ndarray) -> bool | None:
    """Funkcja sprawdzająca czy podana macierz NIE JEST singularna. Przy
    implementacji należy pamiętać o definicji zera maszynowego.

    Args:
        A (np.ndarray): Macierz (n,n) do przetestowania.

    Returns:
        (bool): `True`, jeżeli macierz A nie jest singularna, w przeciwnym 
            wypadku `False`.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not isinstance(A, np.ndarray) or A.ndim != 2 or A.shape[0] != A.shape[1]:
        return None

    det = np.linalg.det(A)

    eps = 1e-12 

    return abs(det) > eps
