from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.types import Numeric, Text, DateTime, DECIMAL, Float
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.dialects import postgresql
from sqlalchemy import func, text

Base = declarative_base()

# use mixins for the shared constraints
# https://docs.sqlalchemy.org/en/13/orm/extensions/declarative/mixins.html

# if there is a lock conflict do this
# https://stackoverflow.com/questions/20373557/postgresql-why-is-drop-view-command-hanging
# select * from pg_locks where not granted;

class ImportMixin(object):
    @declared_attr
    def ImportID(cls):
        return Column(Integer, ForeignKey('imports.ImportID'), nullable=False)


class TimestampMixin(object):
    # calculate this on the server side
    ImportOBIDTS = Column(DateTime(), server_default=func.now(), nullable=False)
    UpdateOBIDTS = Column(DateTime(), onupdate=func.now())


class Import(TimestampMixin, Base):
    __tablename__ = 'imports'
    # these are the import parameters
    ImportID = Column(Integer, primary_key=True)
    User = Column(String(50))
    Protocol = Column(String(50))
    Name = Column(String(50))
    Description = Column(Text)
    StartDTS = Column(DateTime)
    EndDTS = Column(DateTime)
    ChunkSizes = Column(postgresql.JSONB, default=text("'{}'::jsonb"), server_default=text("'{}'::jsonb"))
    # this is the import progress and status
    Progress = Column(postgresql.JSONB, default=text("'{}'::jsonb"), server_default=text("'{}'::jsonb"))
    Status = Column(String(50), default='incomplete')


class Query(Base):
    __tablename__ = 'queries'
    # these are the query ids
    QueryID = Column(String(50), primary_key=True)
    ImportID = Column(Integer, ForeignKey('imports.ImportID'), primary_key=True)


# TODO could add this as foreign key to all the other tables
#  but might make bulk downloads complicated
class PatientId(ImportMixin, TimestampMixin, Base):
    __tablename__ = 'patientids'
    PatientID = Column(String(50), primary_key=True)
    EMPI = Column(DECIMAL(50))
    BWHMRN = Column(DECIMAL(50))
    MGHMRN = Column(DECIMAL(50))
    PMRN = Column(DECIMAL(50))


class Note(ImportMixin, TimestampMixin, Base):
    __tablename__ = 'notes'
    NoteID = Column(String(50), primary_key=True)
    NoteCSNID = Column(DECIMAL(50))
    PatientID = Column(String(50))
    PatientEncounterID = Column(DECIMAL(50))
    NoteTXT = Column(Text())
    DiagnosisNM = Column(Text)
    DiagnosisID = Column(Text)
    CurrentICD10TXT = Column(Text)                      # TODO could use JSONB for list types
    InpatientNoteTypeDSC = Column(String(50))
    EncounterTypeDSC = Column(String(50))
    DepartmentDSC = Column(String(50))
    CurrentAuthorID = Column(String(50))
    ContactDTS = Column(DateTime)
    UpdateDTS = Column(DateTime)
    LastFiledDTS = Column(DateTime)


class EncounterDX(ImportMixin, TimestampMixin, Base):
    __tablename__ = 'encounterdx'
    PatientEncounterID = Column(DECIMAL(50), primary_key=True)
    LineNBR = Column(DECIMAL(10))
    PatientID = Column(String(50))
    EncounterTypeDSC = Column(String(50))
    DepartmentDSC = Column(String(50))
    DiagnosisNM = Column(Text())
    DiagnosisID = Column(DECIMAL(50))
    CurrentICD10TXT = Column(String(50), primary_key=True)
    ContactDTS = Column(DateTime)
    UpdateDTS = Column(DateTime)


class Demographic(ImportMixin, TimestampMixin, Base):
    __tablename__ = 'demographics'
    PatientID = Column(String(50), primary_key=True)
    SexDSC = Column(String(50))
    PatientRaceDSC = Column(String(50))
    EthnicGroupDSC = Column(String(50))
    PatientStatusDSC = Column(String(50))
    BirthDTS = Column(DateTime)
    DeathDTS = Column(DateTime)
    Age = Column(Numeric)                               # TODO this is age at import, maybe calculate in the DB


class Lab(ImportMixin, TimestampMixin, Base):
    __tablename__ = 'labs'
    PatientID = Column(String(50), primary_key=True)
    PatientEncounterID = Column(DECIMAL(50), primary_key=True)
    ResultDTS = Column(DateTime, primary_key=True)
    OrderProcedureID = Column(DECIMAL(50), primary_key=True)
    ComponentID = Column(DECIMAL(50), primary_key=True)
    ComponentNM = Column(Text())
    ComponentCommonNM = Column(Text())
    ComponentTypeDSC = Column(String(50))
    ComponentAbbreviationTXT = Column(String(50))
    BaseNM = Column(String(50))
    ExternalNM = Column(Text())
    ResultTXT = Column(Text(), primary_key=True)        # microbiology culture results are only specified here
    ResultValueNBR = Column(Float)
    InReferenceRangeFLG = Column(String(50))
    ReferenceRangeLowNBR = Column(String(50))
    ReferenceRangeHighNBR = Column(String(50))
    ReferenceRangeUnitCD = Column(String(255))
    DefaultLowCD = Column(String(50))
    DefaultHighCD = Column(String(50))
    DefaultLowFemaleCD = Column(String(50))
    DefaultHighFemaleCD = Column(String(50))
    DefaultUnitCD = Column(String(50))


class Vital(ImportMixin, TimestampMixin, Base):
    __tablename__ = 'vitals'
    PatientID = Column(String(50), primary_key=True)
    RecordedDTS = Column(DateTime, primary_key=True)
    FlowsheetMeasureNM = Column(String(50), primary_key=True)
    DisplayNM = Column(String(50))
    MeasureTXT = Column(String(50))
    MeasureCommentTXT = Column(Text())
    FlowsheetMeasureID = Column(String(50))


class Medication(ImportMixin, TimestampMixin, Base):
    __tablename__ = 'medications'
    PatientID = Column(String(50), primary_key=True)
    PatientEncounterID = Column(DECIMAL(50), primary_key=True)
    OrderID = Column(DECIMAL(50), primary_key=True)
    OriginalMedicationOrderID = Column(DECIMAL(50))
    StartDTS = Column(DateTime)
    EndDTS = Column(DateTime)
    UpdateDTS = Column(DateTime)
    OrderInstantDTS = Column(DateTime)
    OrderStartDTS = Column(DateTime)
    OrderEndDTS = Column(DateTime)
    OrderDiscontinuedDTS = Column(DateTime)
    MedicationID = Column(DECIMAL(50))
    MedicationDSC = Column(Text())
    MedicationDisplayNM = Column(Text())
    SigTXT = Column(Text())
    MedicationRouteDSC = Column(Text())
    PrescriptionQuantityNBR = Column(String(50))
    RefillCNT = Column(String(50))
    DiscreteFrequencyDSC = Column(Text())
    DiscreteDoseAMT = Column(String(50))
    HVDoseUnitDSC = Column(String(50))
    OrderPriorityDSC = Column(String(50))
    OrderStatusDSC = Column(String(50))
    AdditionalInformationOrderStatusDSC = Column(Text())
    OrderClassDSC = Column(String(50))
    ReorderedOrModifiedDSC = Column(String(50))
    MedicationReorderMethodDSC = Column(String(50))
    LastAdministeredDoseCommentTXT = Column(Text())
    MedicationDiscontinueReasonDSC = Column(Text())
    OrderingModeDSC = Column(String(50))
    MedicationPrescribingProviderID = Column(String(50))
    OrderingProviderID = Column(String(50))
    ProviderTypeDSC = Column(String(50))
    ProviderStatusDSC = Column(String(50))
    PatientLocationDSC = Column(String(50))


class Tobacco(ImportMixin, TimestampMixin, Base):
    __tablename__ = 'tobacco'
    PatientID = Column(String(50), primary_key=True)
    ContactDTS = Column(DateTime, primary_key=True)
    TobaccoPacksPerDayCNT = Column(DECIMAL(50))
    TobaccoUsageYearNBR = Column(DECIMAL(50))
    TobaccoCommentTXT = Column(Text())
    TobaccoUserDSC = Column(String(50))
    SmokingTobaccoUseDSC = Column(String(50))
    SmokelessTobaccoUserDSC = Column(String(50))


class MedicalHx(ImportMixin, TimestampMixin, Base):
    __tablename__ = 'medicalhx'
    PatientID = Column(String(50), primary_key=True)
    PatientEncounterID = Column(DECIMAL(50), primary_key=True)
    ICD10CD = Column(String(50), primary_key=True)
    DiagnosisID = Column(DECIMAL(50))
    ContactDTS = Column(DateTime)


# TODO ContactDTS is not the surgical procedure date (issue #29)
class SurgicalHx(ImportMixin, TimestampMixin, Base):
    __tablename__ = 'surgicalhx'
    PatientID = Column(String(50), primary_key=True)
    PatientEncounterID = Column(DECIMAL(50), primary_key=True)
    ProcedureID = Column(DECIMAL(50), primary_key=True)
    ProcedureNM = Column(Text())
    CommentTXT = Column(Text(), primary_key=True)
    ContactDTS = Column(DateTime)


class FamilyHx(ImportMixin, TimestampMixin, Base):
    __tablename__ = 'familyhx'
    PatientID = Column(String(50), primary_key=True)
    PatientEncounterID = Column(DECIMAL(50), primary_key=True)
    MedicalHistoryCD = Column(DECIMAL(50), primary_key=True)
    RelationDSC = Column(String(50), primary_key=True)
    MedicalHistoryDSC = Column(Text())
    ContactDTS = Column(DateTime)


class AdmitDx(ImportMixin, TimestampMixin, Base):
    __tablename__ = 'admitdx'
    PatientID = Column(String(50), primary_key=True)
    PatientEncounterID = Column(DECIMAL(50), primary_key=True)
    DiagnosisID = Column(DECIMAL(50), primary_key=True)
    AdmitDiagnosisTXT = Column(Text())
    ContactDTS = Column(DateTime)


class ProblemList(ImportMixin, TimestampMixin, Base):
    __tablename__ = 'problemlist'
    PatientID = Column(String(50), primary_key=True)
    ICD10CD = Column(String(50), primary_key=True)
    DiagnosisDTS = Column(DateTime, primary_key=True)
    DiagnosisID = Column(DECIMAL(50))
    ProblemDSC = Column(Text())
    ProblemStatusDSC = Column(String(50))


class Hematology(ImportMixin, TimestampMixin, Base):
    __tablename__ = 'hematology'
    PatientID = Column(String(50), primary_key=True)
    PatientEncounterID = Column(DECIMAL(50), primary_key=True)
    ResultDTS = Column(DateTime, primary_key=True)
    RBC = Column(Float)
    WBC = Column(Float)
    HCT = Column(Float)
    HGB = Column(Float)
    MCV = Column(Float)
    RDW = Column(Float)
    MCHC = Column(Float)
    MCH = Column(Float)
    PLT = Column(Float)
    NRBC = Column(Float)
    MPV = Column(Float)


class ReferenceMedication(ImportMixin, TimestampMixin, Base):
    __tablename__ = 'reference_medication'
    MedicationID = Column(DECIMAL(28), primary_key=True)
    MedicationDSC = Column(String(255))
    TherapeuticClassCD = Column(DECIMAL(28))
    TherapeuticClassDSC = Column(String(255))
    PharmaceuticalClassCD = Column(DECIMAL(28))
    PharmaceuticalClassDSC = Column(String(255))
    PharmaceuticalSubclassCD = Column(DECIMAL(28))
    PharmaceuticalSubclassDSC = Column(String(255))
    SimpleGenericCD = Column(String(80))
    SimpleGenericDSC = Column(String(255))
    GenericNM = Column(String(255))
    GenericProductID = Column(String(255))
    StrengthAMT = Column(String(255))
