import numpy as np

import experiment1 as e1
import experiment2 as e2
import ManualStrategy as ms


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "jyu497"


if __name__ == "__main__":
    np.random.seed(902852040)
    ms.run()
    e1.run()
    e2.run()
