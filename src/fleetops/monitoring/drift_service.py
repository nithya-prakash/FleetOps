"""Periodic drift evaluation over recent edge feature samples; creates review recommendations."""

from __future__ import annotations

import logging

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from fleetops.domain.models import utcnow
from fleetops.monitoring.drift import DriftThresholds, evaluate_drift, labelled_quality
from fleetops.monitoring.metrics import ControlPlaneMetrics
from fleetops.persistence.audit import record_audit
from fleetops.persistence.db import session_scope
from fleetops.persistence.tables import DriftReport, FeatureSample, ModelVersion, ReviewRecommendation

logger = logging.getLogger(__name__)


class DriftService:
    def __init__(
        self, sessions: sessionmaker[Session], thresholds: DriftThresholds, window: int, metrics: ControlPlaneMetrics
    ) -> None:
        self._sessions = sessions
        self._th = thresholds
        self._window = window
        self._metrics = metrics

    def run_once(self) -> list[int]:
        """Evaluate every model version that has samples; returns created report ids."""
        ids: list[int] = []
        with session_scope(self._sessions) as db:
            pairs = db.execute(select(FeatureSample.model_name, FeatureSample.model_version).distinct()).all()
        for name, version in pairs:
            rid = self.evaluate(name, version)
            if rid is not None:
                ids.append(rid)
        return ids

    def evaluate(self, name: str, version: int) -> int | None:
        with session_scope(self._sessions) as db:
            mv = db.scalar(select(ModelVersion).where(ModelVersion.name == name, ModelVersion.version == version))
            if mv is None:
                return None
            samples = list(
                db.scalars(
                    select(FeatureSample)
                    .where(FeatureSample.model_name == name, FeatureSample.model_version == version)
                    .order_by(FeatureSample.id.desc())
                    .limit(self._window)
                )
            )
            if not samples:
                return None
            feature_names = [f["name"] for f in mv.input_schema["features"]]
            x = np.array([s.features for s in samples], dtype=np.float64)
            status, results = evaluate_drift(mv.reference_stats, feature_names, x, self._th)
            quality = labelled_quality([s.anomaly for s in samples], [s.label for s in samples])
            quality["note"] = (
                "Drift means input distributions changed; it does not by itself mean the model is inaccurate. "
                "Quality is only measured on samples that carried ground-truth labels."
            )
            report = DriftReport(
                model_name=name,
                model_version=version,
                generated_at=utcnow(),
                window_start=min(s.ts for s in samples),
                window_end=max(s.ts for s in samples),
                sample_count=len(samples),
                status=status,
                features=[r.as_dict() for r in results],
                quality=quality,
                thresholds={
                    "psi_warn": self._th.psi_warn,
                    "psi_alert": self._th.psi_alert,
                    "p_value": self._th.p_value,
                    "min_samples": self._th.min_samples,
                },
            )
            db.add(report)
            db.flush()
            self._metrics.drift_reports.labels(status).inc()
            if status == "ALERT":
                open_rec = db.scalar(
                    select(ReviewRecommendation).where(
                        ReviewRecommendation.model_name == name,
                        ReviewRecommendation.model_version == version,
                        ReviewRecommendation.status == "OPEN",
                    )
                )
                if open_rec is None:
                    flagged = [r.feature for r in results if r.status == "ALERT"]
                    db.add(
                        ReviewRecommendation(
                            drift_report_id=report.id,
                            model_name=name,
                            model_version=version,
                            created_at=utcnow(),
                            message=(
                                f"Input drift ALERT on {', '.join(flagged)} for {name}:{version}. Review whether "
                                "retraining or a new reference is warranted. No automatic deployment was triggered."
                            ),
                        )
                    )
                    record_audit(
                        db, "system", "drift.review_recommended", "model", f"{name}:{version}", features=flagged
                    )
            return report.id
