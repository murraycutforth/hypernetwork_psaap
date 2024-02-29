from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error, normalized_mutual_information

def compute_all_metrics(pred, gt) -> dict:
    assert gt.max() <= 1.0
    assert gt.min() >= -1.0

    return {
        'mse': mean_squared_error(pred, gt),
        'ssim': structural_similarity(pred, gt, data_range=2.0),
        'psnr': peak_signal_noise_ratio(gt, pred),
        'nmi': normalized_mutual_information(pred, gt)
    }
