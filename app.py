import streamlit as st
import pdf2image
import io
from PIL import Image, ImageDraw, ImageFont
import tempfile
from docx import Document
from predict import predict, Predictor, Cfg, PaddleOCR
from fpdf import FPDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def setup_ocr_models():
    # Configure VietOCR
    config = Cfg.load_config_from_name('vgg_transformer')
    config['cnn']['pretrained'] = True
    config['predictor']['beamsearch'] = True
    config['device'] = 'cuda:0'
    recognitor = Predictor(config)
    
    # Configure PaddleOCR
    detector = PaddleOCR(use_angle_cls=False, lang="vi", use_gpu=True)
    
    return recognitor, detector

def pdf_to_images(pdf_file):
    # Convert PDF to images
    return pdf2image.convert_from_bytes(pdf_file.read())

def process_image(image, recognitor, detector):
    # Save image to temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        image.save(tmp.name)
        boxes, texts = predict(recognitor, detector, tmp.name)
    return boxes, texts

def draw_text_on_image(image, boxes, texts):
    # Add padding to the image
    padding = 20  # Adjust padding size as needed
    new_size = (image.width + 2*padding, image.height + 2*padding)
    padded_image = Image.new('RGB', new_size, 'white')
    padded_image.paste(image, (padding, padding))
    
    draw = ImageDraw.Draw(padded_image)
    
    # Try to load a font that supports Vietnamese characters
    try:
        # You can adjust these fonts based on availability on your system
        font_paths = [
            "arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        
        font = None
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, size=20)  # Initial size
                break
            except:
                continue
                
        if font is None:
            font = ImageFont.load_default()
            
    except:
        font = ImageFont.load_default()
    
    # Draw white rectangles and text
    for box, text in zip(boxes, texts):
        # Add padding to box coordinates
        box = [[coord[0] + padding, coord[1] + padding] for coord in box]
        
        # Calculate box dimensions
        box_width = box[1][0] - box[0][0]
        box_height = box[1][1] - box[0][1]
        
        # Estimate appropriate font size based on box height
        font_size = int(box_height * 0.8)  # 80% of box height
        try:
            font = ImageFont.truetype(font.path, size=font_size)
        except:
            pass
        
        # Draw white rectangle
        draw.rectangle([box[0][0], box[0][1], box[1][0], box[1][1]], 
                      fill='white', outline='white')
        
        # Draw text with UTF-8 encoding
        text = text.encode('utf-8').decode('utf-8')  # Ensure proper UTF-8 encoding
        draw.text((box[0][0], box[0][1]), text, fill='black', font=font)
    
    return padded_image

def create_docx(texts):
    doc = Document()
    for text in texts:
        doc.add_paragraph(text)
    return doc

def main():
    st.title("Vietnamese OCR PDF Processor")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Use session state to store processed results
        if 'processed_images' not in st.session_state:
            # Initialize OCR models
            recognitor, detector = setup_ocr_models()
            
            # Convert PDF to images
            images = pdf_to_images(uploaded_file)
            
            # Process each page
            processed_images = []
            all_texts = []
            all_boxes = []
            
            progress_bar = st.progress(0)
            for i, image in enumerate(images):
                # Process image
                boxes, texts = process_image(image, recognitor, detector)
                all_texts.append(texts)
                all_boxes.append(boxes)
                
                # Draw text on image
                # processed_image = draw_text_on_image(image, boxes, texts)
                processed_images.append(image)
                
                # Update progress
                progress_bar.progress((i + 1) / len(images))
            
            # Store results in session state
            st.session_state.processed_images = processed_images
            st.session_state.all_texts = all_texts
            st.session_state.all_boxes = all_boxes
            
            # Register Vietnamese font
            pdfmetrics.registerFont(TTFont('DejaVu', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
            
            pdf_buffer = io.BytesIO()
            c = canvas.Canvas(pdf_buffer, pagesize=A4)
            
            for image, boxes, texts in zip(st.session_state.processed_images, st.session_state.all_boxes, st.session_state.all_texts):
                # Create temporary file for the image
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img:
                    image.save(tmp_img.name, format='PNG')
                    
                    # Calculate scaling factors
                    scale_x = A4[0] / image.width
                    scale_y = A4[1] / image.height
                    
                    # Add page with image
                    c.drawImage(tmp_img.name, 0, 0, width=A4[0], height=A4[1])
                
                # Add selectable text layer
                for box, text in zip(boxes, texts):
                    # Scale box coordinates to PDF size
                    x1 = box[0][0] * scale_x
                    y1 = box[0][1] * scale_y
                    x2 = box[1][0] * scale_x
                    y2 = box[1][1] * scale_y
                    
                    # Calculate scaled dimensions
                    box_width = x2 - x1
                    box_height = y2 - y1
                    
                    # Calculate font size (use a smaller percentage of box height)
                    font_size = int(box_height * 0.8)
                    
                    # Set minimum and maximum font sizes
                    font_size = max(6, min(font_size, 72))  # Prevent too small or too large fonts
                    
                    # Set font and check text width
                    c.setFont('DejaVu', font_size)
                    text_width = c.stringWidth(text, 'DejaVu', font_size)
                    
                    # Adjust font size if text is too wide
                    if text_width > box_width:
                        font_size = int(font_size * (box_width / text_width) * 0.95)
                        font_size = max(6, font_size)  # Ensure minimum font size
                        c.setFont('DejaVu', font_size)
                    
                    # Add padding for the background (in points)
                    padding = 2
                    
                    # Draw white rectangle background
                    c.setFillColorRGB(1, 1, 1, 1)  # Set fill color to white
                    c.rect(
                        x1 - padding,
                        A4[1] - y1 - font_size - padding,
                        box_width + (2 * padding),
                        font_size + (2 * padding),
                        fill=1,
                        stroke=0
                    )
                    
                    # Set text color to black
                    c.setFillColorRGB(0, 0, 0, 1)
                    
                    # Draw the text
                    y_position = A4[1] - y1 - (font_size)
                    c.drawString(x1, y_position, text)
                    
                    # Reset color to black for any subsequent drawing operations
                    # c.setFillColorRGB(0, 0, 0, 1)

                c.showPage()
                
                # Clean up temporary file
                import os
                os.unlink(tmp_img.name)
            
            c.save()
            st.session_state.pdf_buffer = pdf_buffer.getvalue()

        # Export options
        st.subheader("Export Options")
        
        # Direct PDF download
        st.download_button(
            "Download Processed PDF",
            st.session_state.pdf_buffer,
            "processed_document.pdf",
            "application/pdf"
        )
        
        # Direct DOCX download
        doc = create_docx(st.session_state.all_texts)
        docx_buffer = io.BytesIO()
        doc.save(docx_buffer)
        st.download_button(
            "Download DOCX",
            docx_buffer.getvalue(),
            "processed_document.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

if __name__ == "__main__":
    main()