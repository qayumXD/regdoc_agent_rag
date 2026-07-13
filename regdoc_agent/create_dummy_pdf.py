from fpdf import FPDF
import os

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 10, "Software as a Medical Device (SaMD): Clinical Evaluation. \nThis document provides guidance on the clinical evaluation of Software as a Medical Device. \nSection 1: Introduction. \nSaMD must undergo clinical evaluation to demonstrate safety and effectiveness.")
pdf.add_page()
pdf.multi_cell(0, 10, "Section 2: Clinical Evidence. \nValid clinical association is required between the SaMD output and the targeted clinical condition. \nAnalytical validation ensures the SaMD accurately and reliably processes input data.")
pdf.add_page()
pdf.multi_cell(0, 10, "Section 3: Clinical Validation. \nClinical validation is the ability of a SaMD to yield a clinically meaningful output associated with the target use. \nIndependent review may be necessary for high-risk SaMD.")
pdf.output("data/samd_dummy.pdf")
