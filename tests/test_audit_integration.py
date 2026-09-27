import torch
import sys
import os
import logging

# Configure professional logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.audit.auditor import AuditResult
from core.audit.calibrate import calibrate
from core.audit.evidence import classify
from core.audit.attribute import attribute

def generate_mock_audit_result() -> AuditResult:
    """Generates a strictly-typed mock AuditResult conforming to Contract C."""
    return AuditResult(
        sr=torch.rand(4, 256, 256),
        c=torch.rand(256, 256) * 0.05,
        v=torch.rand(256, 256) * 0.05,
        res=torch.rand(4, 1, 64, 64) * 0.04,
        masks=torch.ones(4, 1, 64, 64),
        meta={"dates": ["2026-09-01", "2026-09-06", "2026-09-11", "2026-09-16"]}
    )

def run_integration_test():
    logger.info("Starting integration test for downstream audit modules...")
    
    # 1. Calibration Phase
    logger.info("Executing threshold calibration...")
    try:
        taus = calibrate("abyssos_calibration_data")
        logger.info(f"Calibration successful. Calibrated Thresholds: {taus}")
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        sys.exit(1)

    # 2. Evidence Classification Phase
    logger.info("Executing evidence map classification...")
    ar = generate_mock_audit_result()
    ev_map = classify(ar.c, ar.v, a=None, taus=taus)
    
    assert ev_map.shape == (256, 256), f"Evidence map shape mismatch. Got: {ev_map.shape}"
    logger.info("Evidence map successfully generated (Shape verified: 256x256).")

    # 3. Attribution Phase
    logger.info("Executing object attribution mapping (Target coordinates: x=100, y=100)...")
    attr = attribute(ar, x=100, y=100, taus=taus, radius=8)
    
    logger.info("Attribution Engine Output:")
    logger.info(f"  -> Assigned Evidence Class: {attr['evidence_class']}")
    logger.info(f"  -> Supporting Satellite Passes: {len(attr['supporting_dates'])}")
    logger.info(f"  -> Insufficient/Clouded Passes: {len(attr['missing_dates'])}")
    
    logger.info("Downstream audit integration test passed successfully.")

if __name__ == "__main__":
    run_integration_test()