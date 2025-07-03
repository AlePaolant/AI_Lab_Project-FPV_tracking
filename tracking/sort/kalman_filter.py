import numpy as np

class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        """
        bbox: [x1, y1, x2, y2]
        """
        # Crea stato iniziale
        self.kf = self._init_kalman()
        self.kf['x'][:4] = self._convert_bbox_to_z(bbox)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Aggiorna Kalman con nuova bbox osservata
        """
        z = self._convert_bbox_to_z(bbox)
        self.kf = self._kf_update(self.kf, z)
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1

    def predict(self):
        """
        Predice stato successivo
        """
        self.kf = self._kf_predict(self.kf)
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self._convert_x_to_bbox(self.kf['x']))
        return self.history[-1]

    def get_state(self):
        """
        Ritorna l'ultima bbox stimata
        """
        return self._convert_x_to_bbox(self.kf['x'])

    # --------------------
    # Conversione bbox <-> stato
    def _convert_bbox_to_z(self, bbox):
        """
        [x1,y1,x2,y2] -> [x,y,s,r]
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        s = w * h
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

    def _convert_x_to_bbox(self, x):
        """
        [x,y,s,r] -> [x1,y1,x2,y2]
        """
        s = x[2]
        r = x[3]
        w = np.sqrt(abs(s * r))  # ABS evita NaN
        h = abs(s) / w if w != 0 else 0
        return np.array([x[0] - w/2, x[1] - h/2, x[0] + w/2, x[1] + h/2]).flatten()

    # --------------------
    # Semplice Kalman filter 8D (no librerie esterne)
    def _init_kalman(self):
        kf = {
            'x': np.zeros((8,1)),
            'P': np.eye(8) * 10,
            'F': np.eye(8),
            'Q': np.eye(8) * 0.01,
            'H': np.zeros((4,8)),
            'R': np.eye(4) * 0.1
        }
        for i in range(4):
            kf['F'][i,i+4] = 1
            kf['H'][i,i] = 1
        return kf

    def _kf_predict(self, kf):
        x = np.dot(kf['F'], kf['x'])
        P = np.dot(kf['F'], np.dot(kf['P'], kf['F'].T)) + kf['Q']
        return {**kf, 'x': x, 'P': P}

    def _kf_update(self, kf, z):
        y = z - np.dot(kf['H'], kf['x'])
        S = np.dot(kf['H'], np.dot(kf['P'], kf['H'].T)) + kf['R']
        K = np.dot(kf['P'], np.dot(kf['H'].T, np.linalg.inv(S)))
        x = kf['x'] + np.dot(K, y)
        I = np.eye(8)
        P = np.dot((I - np.dot(K, kf['H'])), kf['P'])
        return {**kf, 'x': x, 'P': P}
