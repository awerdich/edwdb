"""initial import

Revision ID: 60e67101bbe0
Revises: 
Create Date: 2020-05-11 00:00:14.098387

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '60e67101bbe0'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('imports',
    sa.Column('ImportOBIDTS', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.Column('UpdateOBIDTS', sa.DateTime(), nullable=True),
    sa.Column('ImportID', sa.Integer(), nullable=False),
    sa.Column('User', sa.String(length=50), nullable=True),
    sa.Column('Protocol', sa.String(length=50), nullable=True),
    sa.Column('Name', sa.String(length=50), nullable=True),
    sa.Column('Description', sa.Text(), nullable=True),
    sa.Column('StartDTS', sa.DateTime(), nullable=True),
    sa.Column('EndDTS', sa.DateTime(), nullable=True),
    sa.Column('ChunkSizes', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=True),
    sa.Column('Progress', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=True),
    sa.Column('Status', sa.String(length=50), nullable=True),
    sa.PrimaryKeyConstraint('ImportID')
    )
    op.create_table('admitdx',
    sa.Column('ImportOBIDTS', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.Column('UpdateOBIDTS', sa.DateTime(), nullable=True),
    sa.Column('PatientID', sa.String(length=50), nullable=False),
    sa.Column('PatientEncounterID', sa.DECIMAL(precision=50), nullable=False),
    sa.Column('DiagnosisID', sa.DECIMAL(precision=50), nullable=False),
    sa.Column('AdmitDiagnosisTXT', sa.Text(), nullable=True),
    sa.Column('ContactDTS', sa.DateTime(), nullable=True),
    sa.Column('ImportID', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['ImportID'], ['imports.ImportID'], ),
    sa.PrimaryKeyConstraint('PatientID', 'PatientEncounterID', 'DiagnosisID')
    )
    op.create_table('demographics',
    sa.Column('ImportOBIDTS', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.Column('UpdateOBIDTS', sa.DateTime(), nullable=True),
    sa.Column('PatientID', sa.String(length=50), nullable=False),
    sa.Column('SexDSC', sa.String(length=50), nullable=True),
    sa.Column('PatientRaceDSC', sa.String(length=50), nullable=True),
    sa.Column('EthnicGroupDSC', sa.String(length=50), nullable=True),
    sa.Column('PatientStatusDSC', sa.String(length=50), nullable=True),
    sa.Column('BirthDTS', sa.DateTime(), nullable=True),
    sa.Column('DeathDTS', sa.DateTime(), nullable=True),
    sa.Column('Age', sa.Numeric(), nullable=True),
    sa.Column('ImportID', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['ImportID'], ['imports.ImportID'], ),
    sa.PrimaryKeyConstraint('PatientID')
    )
    op.create_table('encounterdx',
    sa.Column('ImportOBIDTS', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.Column('UpdateOBIDTS', sa.DateTime(), nullable=True),
    sa.Column('PatientEncounterID', sa.DECIMAL(precision=50), nullable=False),
    sa.Column('LineNBR', sa.DECIMAL(precision=10), nullable=True),
    sa.Column('PatientID', sa.String(length=50), nullable=True),
    sa.Column('EncounterTypeDSC', sa.String(length=50), nullable=True),
    sa.Column('DepartmentDSC', sa.String(length=50), nullable=True),
    sa.Column('DiagnosisNM', sa.Text(), nullable=True),
    sa.Column('DiagnosisID', sa.DECIMAL(precision=50), nullable=True),
    sa.Column('CurrentICD10TXT', sa.String(length=50), nullable=False),
    sa.Column('ContactDTS', sa.DateTime(), nullable=True),
    sa.Column('UpdateDTS', sa.DateTime(), nullable=True),
    sa.Column('ImportID', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['ImportID'], ['imports.ImportID'], ),
    sa.PrimaryKeyConstraint('PatientEncounterID', 'CurrentICD10TXT')
    )
    op.create_table('familyhx',
    sa.Column('ImportOBIDTS', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.Column('UpdateOBIDTS', sa.DateTime(), nullable=True),
    sa.Column('PatientID', sa.String(length=50), nullable=False),
    sa.Column('PatientEncounterID', sa.DECIMAL(precision=50), nullable=False),
    sa.Column('MedicalHistoryCD', sa.DECIMAL(precision=50), nullable=False),
    sa.Column('RelationDSC', sa.String(length=50), nullable=False),
    sa.Column('MedicalHistoryDSC', sa.Text(), nullable=True),
    sa.Column('ContactDTS', sa.DateTime(), nullable=True),
    sa.Column('ImportID', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['ImportID'], ['imports.ImportID'], ),
    sa.PrimaryKeyConstraint('PatientID', 'PatientEncounterID', 'MedicalHistoryCD', 'RelationDSC')
    )
    op.create_table('hematology',
    sa.Column('ImportOBIDTS', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.Column('UpdateOBIDTS', sa.DateTime(), nullable=True),
    sa.Column('PatientID', sa.String(length=50), nullable=False),
    sa.Column('PatientEncounterID', sa.DECIMAL(precision=50), nullable=False),
    sa.Column('ResultDTS', sa.DateTime(), nullable=False),
    sa.Column('RBC', sa.Float(), nullable=True),
    sa.Column('WBC', sa.Float(), nullable=True),
    sa.Column('HCT', sa.Float(), nullable=True),
    sa.Column('HGB', sa.Float(), nullable=True),
    sa.Column('MCV', sa.Float(), nullable=True),
    sa.Column('RDW', sa.Float(), nullable=True),
    sa.Column('MCHC', sa.Float(), nullable=True),
    sa.Column('MCH', sa.Float(), nullable=True),
    sa.Column('PLT', sa.Float(), nullable=True),
    sa.Column('NRBC', sa.Float(), nullable=True),
    sa.Column('MPV', sa.Float(), nullable=True),
    sa.Column('ImportID', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['ImportID'], ['imports.ImportID'], ),
    sa.PrimaryKeyConstraint('PatientID', 'PatientEncounterID', 'ResultDTS')
    )
    op.create_table('labs',
    sa.Column('ImportOBIDTS', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.Column('UpdateOBIDTS', sa.DateTime(), nullable=True),
    sa.Column('PatientID', sa.String(length=50), nullable=False),
    sa.Column('PatientEncounterID', sa.DECIMAL(precision=50), nullable=False),
    sa.Column('ResultDTS', sa.DateTime(), nullable=False),
    sa.Column('OrderProcedureID', sa.DECIMAL(precision=50), nullable=False),
    sa.Column('ComponentID', sa.DECIMAL(precision=50), nullable=False),
    sa.Column('ComponentNM', sa.Text(), nullable=True),
    sa.Column('ComponentCommonNM', sa.Text(), nullable=True),
    sa.Column('ComponentTypeDSC', sa.String(length=50), nullable=True),
    sa.Column('ComponentAbbreviationTXT', sa.String(length=50), nullable=True),
    sa.Column('BaseNM', sa.String(length=50), nullable=True),
    sa.Column('ExternalNM', sa.Text(), nullable=True),
    sa.Column('ResultTXT', sa.Text(), nullable=False),
    sa.Column('ResultValueNBR', sa.Float(), nullable=True),
    sa.Column('InReferenceRangeFLG', sa.String(length=50), nullable=True),
    sa.Column('ReferenceRangeLowNBR', sa.String(length=50), nullable=True),
    sa.Column('ReferenceRangeHighNBR', sa.String(length=50), nullable=True),
    sa.Column('ReferenceRangeUnitCD', sa.String(length=255), nullable=True),
    sa.Column('DefaultLowCD', sa.String(length=50), nullable=True),
    sa.Column('DefaultHighCD', sa.String(length=50), nullable=True),
    sa.Column('DefaultLowFemaleCD', sa.String(length=50), nullable=True),
    sa.Column('DefaultHighFemaleCD', sa.String(length=50), nullable=True),
    sa.Column('DefaultUnitCD', sa.String(length=50), nullable=True),
    sa.Column('ImportID', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['ImportID'], ['imports.ImportID'], ),
    sa.PrimaryKeyConstraint('PatientID', 'PatientEncounterID', 'ResultDTS', 'OrderProcedureID', 'ComponentID', 'ResultTXT')
    )
    op.create_table('medicalhx',
    sa.Column('ImportOBIDTS', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.Column('UpdateOBIDTS', sa.DateTime(), nullable=True),
    sa.Column('PatientID', sa.String(length=50), nullable=False),
    sa.Column('PatientEncounterID', sa.DECIMAL(precision=50), nullable=False),
    sa.Column('ICD10CD', sa.String(length=50), nullable=False),
    sa.Column('DiagnosisID', sa.DECIMAL(precision=50), nullable=True),
    sa.Column('ContactDTS', sa.DateTime(), nullable=True),
    sa.Column('ImportID', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['ImportID'], ['imports.ImportID'], ),
    sa.PrimaryKeyConstraint('PatientID', 'PatientEncounterID', 'ICD10CD')
    )
    op.create_table('medications',
    sa.Column('ImportOBIDTS', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.Column('UpdateOBIDTS', sa.DateTime(), nullable=True),
    sa.Column('PatientID', sa.String(length=50), nullable=False),
    sa.Column('PatientEncounterID', sa.DECIMAL(precision=50), nullable=False),
    sa.Column('OrderID', sa.DECIMAL(precision=50), nullable=False),
    sa.Column('OriginalMedicationOrderID', sa.DECIMAL(precision=50), nullable=True),
    sa.Column('StartDTS', sa.DateTime(), nullable=True),
    sa.Column('EndDTS', sa.DateTime(), nullable=True),
    sa.Column('UpdateDTS', sa.DateTime(), nullable=True),
    sa.Column('OrderInstantDTS', sa.DateTime(), nullable=True),
    sa.Column('OrderStartDTS', sa.DateTime(), nullable=True),
    sa.Column('OrderEndDTS', sa.DateTime(), nullable=True),
    sa.Column('OrderDiscontinuedDTS', sa.DateTime(), nullable=True),
    sa.Column('MedicationID', sa.DECIMAL(precision=50), nullable=True),
    sa.Column('MedicationDSC', sa.Text(), nullable=True),
    sa.Column('MedicationDisplayNM', sa.Text(), nullable=True),
    sa.Column('SigTXT', sa.Text(), nullable=True),
    sa.Column('MedicationRouteDSC', sa.Text(), nullable=True),
    sa.Column('PrescriptionQuantityNBR', sa.String(length=50), nullable=True),
    sa.Column('RefillCNT', sa.String(length=50), nullable=True),
    sa.Column('DiscreteFrequencyDSC', sa.Text(), nullable=True),
    sa.Column('DiscreteDoseAMT', sa.String(length=50), nullable=True),
    sa.Column('HVDoseUnitDSC', sa.String(length=50), nullable=True),
    sa.Column('OrderPriorityDSC', sa.String(length=50), nullable=True),
    sa.Column('OrderStatusDSC', sa.String(length=50), nullable=True),
    sa.Column('AdditionalInformationOrderStatusDSC', sa.Text(), nullable=True),
    sa.Column('OrderClassDSC', sa.String(length=50), nullable=True),
    sa.Column('ReorderedOrModifiedDSC', sa.String(length=50), nullable=True),
    sa.Column('MedicationReorderMethodDSC', sa.String(length=50), nullable=True),
    sa.Column('LastAdministeredDoseCommentTXT', sa.Text(), nullable=True),
    sa.Column('MedicationDiscontinueReasonDSC', sa.Text(), nullable=True),
    sa.Column('OrderingModeDSC', sa.String(length=50), nullable=True),
    sa.Column('MedicationPrescribingProviderID', sa.String(length=50), nullable=True),
    sa.Column('OrderingProviderID', sa.String(length=50), nullable=True),
    sa.Column('ProviderTypeDSC', sa.String(length=50), nullable=True),
    sa.Column('ProviderStatusDSC', sa.String(length=50), nullable=True),
    sa.Column('PatientLocationDSC', sa.String(length=50), nullable=True),
    sa.Column('ImportID', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['ImportID'], ['imports.ImportID'], ),
    sa.PrimaryKeyConstraint('PatientID', 'PatientEncounterID', 'OrderID')
    )
    op.create_table('notes',
    sa.Column('ImportOBIDTS', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.Column('UpdateOBIDTS', sa.DateTime(), nullable=True),
    sa.Column('NoteID', sa.String(length=50), nullable=False),
    sa.Column('NoteCSNID', sa.DECIMAL(precision=50), nullable=True),
    sa.Column('PatientID', sa.String(length=50), nullable=True),
    sa.Column('PatientEncounterID', sa.DECIMAL(precision=50), nullable=True),
    sa.Column('NoteTXT', sa.Text(), nullable=True),
    sa.Column('DiagnosisNM', sa.Text(), nullable=True),
    sa.Column('DiagnosisID', sa.Text(), nullable=True),
    sa.Column('CurrentICD10TXT', sa.Text(), nullable=True),
    sa.Column('InpatientNoteTypeDSC', sa.String(length=50), nullable=True),
    sa.Column('EncounterTypeDSC', sa.String(length=50), nullable=True),
    sa.Column('DepartmentDSC', sa.String(length=50), nullable=True),
    sa.Column('CurrentAuthorID', sa.String(length=50), nullable=True),
    sa.Column('ContactDTS', sa.DateTime(), nullable=True),
    sa.Column('UpdateDTS', sa.DateTime(), nullable=True),
    sa.Column('LastFiledDTS', sa.DateTime(), nullable=True),
    sa.Column('ImportID', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['ImportID'], ['imports.ImportID'], ),
    sa.PrimaryKeyConstraint('NoteID')
    )
    op.create_table('patientids',
    sa.Column('ImportOBIDTS', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.Column('UpdateOBIDTS', sa.DateTime(), nullable=True),
    sa.Column('PatientID', sa.String(length=50), nullable=False),
    sa.Column('EMPI', sa.DECIMAL(precision=50), nullable=True),
    sa.Column('BWHMRN', sa.DECIMAL(precision=50), nullable=True),
    sa.Column('MGHMRN', sa.DECIMAL(precision=50), nullable=True),
    sa.Column('PMRN', sa.DECIMAL(precision=50), nullable=True),
    sa.Column('ImportID', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['ImportID'], ['imports.ImportID'], ),
    sa.PrimaryKeyConstraint('PatientID')
    )
    op.create_table('problemlist',
    sa.Column('ImportOBIDTS', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.Column('UpdateOBIDTS', sa.DateTime(), nullable=True),
    sa.Column('PatientID', sa.String(length=50), nullable=False),
    sa.Column('ICD10CD', sa.String(length=50), nullable=False),
    sa.Column('DiagnosisDTS', sa.DateTime(), nullable=False),
    sa.Column('DiagnosisID', sa.DECIMAL(precision=50), nullable=True),
    sa.Column('ProblemDSC', sa.Text(), nullable=True),
    sa.Column('ProblemStatusDSC', sa.String(length=50), nullable=True),
    sa.Column('ImportID', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['ImportID'], ['imports.ImportID'], ),
    sa.PrimaryKeyConstraint('PatientID', 'ICD10CD', 'DiagnosisDTS')
    )
    op.create_table('queries',
    sa.Column('QueryID', sa.String(length=50), nullable=False),
    sa.Column('ImportID', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['ImportID'], ['imports.ImportID'], ),
    sa.PrimaryKeyConstraint('QueryID', 'ImportID')
    )
    op.create_table('reference_medication',
    sa.Column('ImportOBIDTS', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.Column('UpdateOBIDTS', sa.DateTime(), nullable=True),
    sa.Column('MedicationID', sa.DECIMAL(precision=28), nullable=False),
    sa.Column('MedicationDSC', sa.String(length=255), nullable=True),
    sa.Column('TherapeuticClassCD', sa.DECIMAL(precision=28), nullable=True),
    sa.Column('TherapeuticClassDSC', sa.String(length=255), nullable=True),
    sa.Column('PharmaceuticalClassCD', sa.DECIMAL(precision=28), nullable=True),
    sa.Column('PharmaceuticalClassDSC', sa.String(length=255), nullable=True),
    sa.Column('PharmaceuticalSubclassCD', sa.DECIMAL(precision=28), nullable=True),
    sa.Column('PharmaceuticalSubclassDSC', sa.String(length=255), nullable=True),
    sa.Column('SimpleGenericCD', sa.String(length=80), nullable=True),
    sa.Column('SimpleGenericDSC', sa.String(length=255), nullable=True),
    sa.Column('GenericNM', sa.String(length=255), nullable=True),
    sa.Column('GenericProductID', sa.String(length=255), nullable=True),
    sa.Column('StrengthAMT', sa.String(length=255), nullable=True),
    sa.Column('ImportID', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['ImportID'], ['imports.ImportID'], ),
    sa.PrimaryKeyConstraint('MedicationID')
    )
    op.create_table('surgicalhx',
    sa.Column('ImportOBIDTS', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.Column('UpdateOBIDTS', sa.DateTime(), nullable=True),
    sa.Column('PatientID', sa.String(length=50), nullable=False),
    sa.Column('PatientEncounterID', sa.DECIMAL(precision=50), nullable=False),
    sa.Column('ProcedureID', sa.DECIMAL(precision=50), nullable=False),
    sa.Column('ProcedureNM', sa.Text(), nullable=True),
    sa.Column('CommentTXT', sa.Text(), nullable=False),
    sa.Column('ContactDTS', sa.DateTime(), nullable=True),
    sa.Column('ImportID', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['ImportID'], ['imports.ImportID'], ),
    sa.PrimaryKeyConstraint('PatientID', 'PatientEncounterID', 'ProcedureID', 'CommentTXT')
    )
    op.create_table('tobacco',
    sa.Column('ImportOBIDTS', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.Column('UpdateOBIDTS', sa.DateTime(), nullable=True),
    sa.Column('PatientID', sa.String(length=50), nullable=False),
    sa.Column('ContactDTS', sa.DateTime(), nullable=False),
    sa.Column('TobaccoPacksPerDayCNT', sa.DECIMAL(precision=50), nullable=True),
    sa.Column('TobaccoUsageYearNBR', sa.DECIMAL(precision=50), nullable=True),
    sa.Column('TobaccoCommentTXT', sa.Text(), nullable=True),
    sa.Column('TobaccoUserDSC', sa.String(length=50), nullable=True),
    sa.Column('SmokingTobaccoUseDSC', sa.String(length=50), nullable=True),
    sa.Column('SmokelessTobaccoUserDSC', sa.String(length=50), nullable=True),
    sa.Column('ImportID', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['ImportID'], ['imports.ImportID'], ),
    sa.PrimaryKeyConstraint('PatientID', 'ContactDTS')
    )
    op.create_table('vitals',
    sa.Column('ImportOBIDTS', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.Column('UpdateOBIDTS', sa.DateTime(), nullable=True),
    sa.Column('PatientID', sa.String(length=50), nullable=False),
    sa.Column('RecordedDTS', sa.DateTime(), nullable=False),
    sa.Column('FlowsheetMeasureNM', sa.String(length=50), nullable=False),
    sa.Column('DisplayNM', sa.String(length=50), nullable=True),
    sa.Column('MeasureTXT', sa.String(length=50), nullable=True),
    sa.Column('MeasureCommentTXT', sa.Text(), nullable=True),
    sa.Column('FlowsheetMeasureID', sa.String(length=50), nullable=True),
    sa.Column('ImportID', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['ImportID'], ['imports.ImportID'], ),
    sa.PrimaryKeyConstraint('PatientID', 'RecordedDTS', 'FlowsheetMeasureNM')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('vitals')
    op.drop_table('tobacco')
    op.drop_table('surgicalhx')
    op.drop_table('reference_medication')
    op.drop_table('queries')
    op.drop_table('problemlist')
    op.drop_table('patientids')
    op.drop_table('notes')
    op.drop_table('medications')
    op.drop_table('medicalhx')
    op.drop_table('labs')
    op.drop_table('hematology')
    op.drop_table('familyhx')
    op.drop_table('encounterdx')
    op.drop_table('demographics')
    op.drop_table('admitdx')
    op.drop_table('imports')
    # ### end Alembic commands ###