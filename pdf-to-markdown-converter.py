import os
import fitz  # PyMuPDF
import re
import shutil
from pathlib import Path

def convert_pdf_to_markdown(pdf_path, output_path):
    """
    Converte un file PDF in markdown e lo salva nel percorso specificato.
    
    Args:
        pdf_path: Percorso del file PDF da convertire
        output_path: Percorso dove salvare il file markdown risultante
    """
    # Apri il documento PDF
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Errore nell'apertura del file {pdf_path}: {e}")
        return False
    
    # Crea una stringa dove memorizzare il contenuto markdown
    markdown_content = ""
    
    # Estrai il testo da ogni pagina e formattalo
    for page_num, page in enumerate(doc):
        # Estrai il testo dalla pagina
        text = page.get_text()
        
        # Se è la prima pagina, cerca di estrarre un titolo
        if page_num == 0:
            # Dividi in righe e cerca una riga che sembri un titolo
            lines = text.split('\n')
            if lines and len(lines[0].strip()) > 0:
                markdown_content += f"# {lines[0].strip()}\n\n"
                # Rimuovi il titolo dal testo per evitare duplicazione
                text = '\n'.join(lines[1:])
        
        # Aggiungi il testo della pagina al markdown
        markdown_content += text + "\n\n"
        
        # Aggiungi un separatore tra le pagine (tranne l'ultima)
        if page_num < len(doc) - 1:
            markdown_content += "---\n\n"
    
    # Chiudi il documento
    doc.close()
    
    # Pulisci il testo da caratteri problematici e formatta meglio
    # Rimuovi spazi multipli
    markdown_content = re.sub(r' +', ' ', markdown_content)
    # Rimuovi interruzioni di riga multiple
    markdown_content = re.sub(r'\n{3,}', '\n\n', markdown_content)
    
    # Scrivi il contenuto nel file markdown
    try:
        with open(output_path, 'w', encoding='utf-8') as md_file:
            md_file.write(markdown_content)
        return True
    except Exception as e:
        print(f"Errore nella scrittura del file {output_path}: {e}")
        return False

def main():
    # Definisci i percorsi
    input_folder = "Psicogiuridico"
    output_folder = "pdf_md"
    
    # Assicurati che i percorsi siano assoluti
    input_folder = os.path.abspath(input_folder)
    output_folder = os.path.abspath(output_folder)
    
    # Verifica se la cartella di input esiste
    if not os.path.exists(input_folder):
        print(f"La cartella {input_folder} non esiste.")
        return
    
    # Crea la cartella di output se non esiste
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Creata la cartella {output_folder}")
    
    # Cerca tutti i file PDF nella cartella di input
    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"Nessun file PDF trovato in {input_folder}")
        return
    
    print(f"Trovati {len(pdf_files)} file PDF da convertire.")
    
    # Converti ogni file PDF in markdown
    successful_conversions = 0
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_folder, pdf_file)
        # Crea il nome del file markdown sostituendo l'estensione .pdf con .md
        md_filename = os.path.splitext(pdf_file)[0] + '.md'
        md_path = os.path.join(output_folder, md_filename)
        
        print(f"Conversione di {pdf_file} in {md_filename}...")
        
        if convert_pdf_to_markdown(pdf_path, md_path):
            successful_conversions += 1
            print(f"Conversione completata con successo: {md_filename}")
        else:
            print(f"Conversione fallita: {pdf_file}")
    
    print(f"\nConversione completata. {successful_conversions} file su {len(pdf_files)} convertiti con successo.")

if __name__ == "__main__":
    main()
