# Imports

import os
from sqlalchemy import Table, MetaData
from sqlalchemy.schema import ForeignKeyConstraint
from sqlalchemy.orm import relationship


def tables_to_meta(engine):
    ''' Set up the tables using the connections defined by the engine '''

    # The access variable should be in the environment when this function is called
    access = os.environ['ACCESS']

    # produce our own MetaData object
    meta = MetaData()

    # Reference tables
    Table('IdentityType', meta, schema = 'Reference', autoload = True, autoload_with = engine)
    Table('Medication', meta, schema = 'Reference', autoload = True, autoload_with = engine)

    # Patient tables
    Table('Identity_' + access, meta, schema ='Patient', autoload = True, autoload_with = engine)

    # Clinical notes tables
    Table('Note_' + access, meta, schema ='Clinical', autoload = True, autoload_with = engine)
    Table('NoteText_' + access, meta, schema ='Clinical', autoload = True, autoload_with = engine)

    return meta

def patient_identity_class(Base):

    access = os.environ['ACCESS']

    class PatientIdentity(Base):
        tab = Base.metadata.tables['Patient.Identity_' + access]
        __table__ = tab
        __mapper_args__ = {
            'primary_key': [tab.c.PatientID, tab.c.LineNBR]
        }
    return PatientIdentity

def clinical_note_classes(Base):
    access = os.environ['ACCESS']
    class ClinicalNote(Base):
        tab = Base.metadata.tables['Clinical.Note_'+access]
        __table__ = tab
        __mapper_args__ = {
            'primary_key': tab.c.NoteID
        }
        note_texts = relationship('ClinicalNoteText', back_populates='note')

    class ClinicalNoteText(Base):
        tab = Base.metadata.tables['Clinical.NoteText_'+access]
        tab_note = Base.metadata.tables['Clinical.Note_'+access]
        __table__ = tab
        __mapper_args__ = {
            'primary_key': [tab.c.NoteCSNID, tab.c.LineNBR]
        }
        __table_args__ = (
            ForeignKeyConstraint([tab.c.NoteID], [tab_note.c.NoteID]),
        )
        note = relationship('ClinicalNote', back_populates = 'note_texts')

    return ClinicalNote, ClinicalNoteText
