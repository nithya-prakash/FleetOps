"""model evaluation metadata and deployment quality report

Revision ID: 0002
Revises: 0001
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("model_versions") as batch:
        # Existing rows get {}: they have no comparable evaluation, so a gate fails safely on them.
        batch.add_column(sa.Column("evaluation", sa.JSON(), nullable=False, server_default="{}"))
    with op.batch_alter_table("deployments") as batch:
        batch.add_column(sa.Column("quality_report", sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("deployments") as batch:
        batch.drop_column("quality_report")
    with op.batch_alter_table("model_versions") as batch:
        batch.drop_column("evaluation")
