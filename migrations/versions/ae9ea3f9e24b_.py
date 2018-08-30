"""empty message

Revision ID: ae9ea3f9e24b
Revises: 9c72079e7309
Create Date: 2018-08-30 14:41:03.954105

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'ae9ea3f9e24b'
down_revision = '9c72079e7309'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('mahasiswa', sa.Column('lulus', sa.Boolean(), nullable=True))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('mahasiswa', 'lulus')
    # ### end Alembic commands ###
