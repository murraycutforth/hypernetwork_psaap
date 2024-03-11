from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error, normalized_mutual_information

def compute_all_metrics(pred, gt, data_range=2.0) -> dict:
    assert gt.max() <= 3.0
    assert gt.min() >= -3.0

    return {
        'mse': mean_squared_error(pred, gt),
        'ssim': structural_similarity(pred, gt, data_range=data_range),
        'psnr': peak_signal_noise_ratio(gt, pred, data_range=data_range),
        'nmi': normalized_mutual_information(pred, gt)
    }
