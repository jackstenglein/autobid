import os
from PyPDF4 import PdfFileReader, PdfFileWriter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.colors import white
from reportlab.lib.pagesizes import letter

""" Overlays each page of the foreground PDF above the first page of the background PDF
    and writes the result to a new output PDF. 

    Each parameter is the string filename of the respective PDF.
"""
def overlay_pdfs(foreground_filename, background_filename, output_filename):

    background_reader = PdfFileReader(background_filename)
    foreground_reader = PdfFileReader(foreground_filename)
    output_writer = PdfFileWriter()

    # Overlay all the background pages
    page = 0
    while page < background_reader.getNumPages() and page < foreground_reader.getNumPages():
        background_page = background_reader.getPage(page)
        foreground_page = foreground_reader.getPage(page)
        background_page.mergePage(foreground_page)
        output_writer.addPage(background_page)
        page += 1

    # Add the remaining foreground pages
    while page < foreground_reader.getNumPages():
        output_writer.addPage(foreground_reader.getPage(page))
        page += 1

    # Write the final PDF
    with open(output_filename, 'wb') as out:
        output_writer.write(out)

""" Creates a PDF to embed in the background of other PDFs using the given list of words. """
def create_background_pdf(background_words, output_filename):
    styles = getSampleStyleSheet()
    white_style = ParagraphStyle('white', parent=styles['Normal'], textColor=white)
    story = []
    doc = SimpleDocTemplate(output_filename, pagesize=letter)

    # Repeat background words to use the full page
    text_content = ' '.join(background_words)
    P = Paragraph(text_content, white_style)
    story.append(P)
    doc.build(story,)

""" Embeds the given word list into each page of the original PDF file. The result is written
    to the specified output file.
"""
def embed_words(words, original_file, output_file):
    overlay_file = 'overlay.pdf'
    create_background_pdf(words, overlay_file)
    overlay_pdfs(original_file, overlay_file, output_file)
    try:
        os.remove(overlay_file)
    except:
        pass


def main():
    embed_words(['testing', 'automatic', 'embedding', 'asdf', 'jkl']*15, '../original.pdf', '../output.pdf')

if __name__ == '__main__':
    main()