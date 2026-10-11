"""elimina la tabla predictions (sin uso: los GET de predicciones ya no escriben)

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-11 12:00:00
"""
import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index(op.f("ix_predictions_ticker"), table_name="predictions")
    op.drop_table("predictions")


def downgrade() -> None:
    # Recrea la tabla exactamente como la definía 0001
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("ticker", sa.String(length=16), nullable=False),
        sa.Column("as_of_date", sa.Date(), nullable=False),
        sa.Column("probability", sa.Float(), nullable=False),
        sa.Column("predicted_class", sa.Integer(), nullable=False),
        sa.Column("model_version", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["ticker"], ["tickers.symbol"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("ticker", "as_of_date", "model_version", name="uq_prediction"),
    )
    op.create_index(op.f("ix_predictions_ticker"), "predictions", ["ticker"], unique=False)