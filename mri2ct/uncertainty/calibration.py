import torch
import numpy as np
from sklearn.isotonic import IsotonicRegression
from scipy import stats
class UncertaintyCalibrator:
    def __init__(self):
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.is_fitted = False
    def fit(self, uncertainties, errors):
        uncertainties = np.array(uncertainties).flatten()
        errors = np.array(errors).flatten()
        mask = np.isfinite(uncertainties) & np.isfinite(errors)
        uncertainties = uncertainties[mask]
        errors = errors[mask]
        if len(uncertainties) > 0:
            self.calibrator.fit(uncertainties, errors)
            self.is_fitted = True
    def calibrate(self, uncertainties):
        if not self.is_fitted:
            return uncertainties
        original_shape = uncertainties.shape
        uncertainties_flat = uncertainties.flatten()
        calibrated = self.calibrator.predict(uncertainties_flat)
        calibrated = calibrated.reshape(original_shape)
        return calibrated
    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.calibrator, f)
    def load(self, path):
        import pickle
        with open(path, 'rb') as f:
            self.calibrator = pickle.load(f)
        self.is_fitted = True
def compute_calibration_metrics(uncertainties, errors, num_bins=10):
    uncertainties = np.array(uncertainties).flatten()
    errors = np.array(errors).flatten()
    mask = np.isfinite(uncertainties) & np.isfinite(errors)
    uncertainties = uncertainties[mask]
    errors = errors[mask]
    sorted_indices = np.argsort(uncertainties)
    uncertainties = uncertainties[sorted_indices]
    errors = errors[sorted_indices]
    bin_size = len(uncertainties) // num_bins
    calibration_x = []
    calibration_y = []
    for i in range(num_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < num_bins - 1 else len(uncertainties)
        bin_uncertainty = uncertainties[start:end].mean()
        bin_error = errors[start:end].mean()
        calibration_x.append(bin_uncertainty)
        calibration_y.append(bin_error)
    calibration_x = np.array(calibration_x)
    calibration_y = np.array(calibration_y)
    ece = np.mean(np.abs(calibration_x - calibration_y))
    correlation, _ = stats.pearsonr(uncertainties, errors)
    return {
        'ece': ece,
        'correlation': correlation,
        'calibration_curve': (calibration_x, calibration_y)
    }
if __name__ == "__main__":
    np.random.seed(42)
    uncertainties = np.random.rand(1000) * 2
    errors = uncertainties + np.random.randn(1000) * 0.3
    errors = np.abs(errors)
    train_size = 800
    train_unc = uncertainties[:train_size]
    train_err = errors[:train_size]
    test_unc = uncertainties[train_size:]
    test_err = errors[train_size:]
    calibrator = UncertaintyCalibrator()
    calibrator.fit(train_unc, train_err)
    calibrated_unc = calibrator.calibrate(test_unc)
    metrics_before = compute_calibration_metrics(test_unc, test_err)
    metrics_after = compute_calibration_metrics(calibrated_unc, test_err)
    print(f"Before calibration - ECE: {metrics_before['ece']:.4f}, "
          f"Correlation: {metrics_before['correlation']:.4f}")
    print(f"After calibration - ECE: {metrics_after['ece']:.4f}, "
          f"Correlation: {metrics_after['correlation']:.4f}")
