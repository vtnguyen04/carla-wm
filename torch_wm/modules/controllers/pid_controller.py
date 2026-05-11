"""Simple PID Controller for expert data collection"""

from collections import deque


class PIDController:
    """Simple PID Controller"""

    def __init__(self, Kp=0.6, Ki=0.01, Kd=0.15, dt=0.03):
        self._kp = Kp
        self._ki = Ki
        self._kd = Kd
        self._dt = dt
        self._error_buffer = deque(maxlen=20)  # Tăng buffer để cua siêu mượt

    def run(self, error):
        """
        Execute PID control step

        :param error: control error
        :return: control output
        """
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return self._kp * error + self._kd * _de + self._ki * _ie

