import os
import zipfile
import json
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

class Exporter:
    def __init__(self, reports_dir, models_dir):
        self.reports_dir = reports_dir
        self.models_dir = models_dir

    def _generate_pdf_report(self, df_clean, activity_log, metrics):
        pdf_path = os.path.join(self.reports_dir, "report.pdf")
        styles = getSampleStyleSheet()
        story = []

        # Title
        story.append(Paragraph("<b>📊 DataForge Report</b>", styles['Title']))
        story.append(Spacer(1, 20))

        # Dataset info
        story.append(Paragraph("<b>Dataset Information</b>", styles['Heading2']))
        story.append(Paragraph(f"Rows: {df_clean.shape[0]}, Columns: {df_clean.shape[1]}", styles['Normal']))
        story.append(Spacer(1, 12))

        # Cleaning steps
        if "cleaning_steps" in activity_log:
            story.append(Paragraph("<b>Cleaning Steps</b>", styles['Heading2']))
            for step in activity_log["cleaning_steps"]:
                story.append(Paragraph(f"• {step}", styles['Normal']))
            story.append(Spacer(1, 12))

        # Visualizations
        if "visualizations" in activity_log:
            story.append(Paragraph("<b>Visualizations</b>", styles['Heading2']))
            for viz in activity_log["visualizations"]:
                if isinstance(viz, dict):  # expected format: {name, path}
                    story.append(Paragraph(f"• {viz.get('name', 'Visualization')}", styles['Normal']))
                    if viz.get("path") and os.path.exists(viz["path"]):
                        story.append(Image(viz["path"], width=400, height=250))
                else:
                    story.append(Paragraph(f"• {viz}", styles['Normal']))
            story.append(Spacer(1, 12))

        # Model training
        if "model_training" in activity_log:
            story.append(Paragraph("<b>Model Training</b>", styles['Heading2']))

            model_trainings = activity_log["model_training"]

            # Normalize input
            if isinstance(model_trainings, dict):
                model_trainings = [model_trainings]
            elif isinstance(model_trainings, str):
                model_trainings = [model_trainings]

            for mt in model_trainings:
                if isinstance(mt, dict):
                    story.append(Paragraph(f"Target Column: {mt.get('target', 'N/A')}", styles['Normal']))
                    story.append(Paragraph(f"Chosen Model: {mt.get('chosen_model', 'N/A')}", styles['Normal']))
                    story.append(Paragraph(f"Train Split: {mt.get('train_split', 'N/A')}, Test Split: {mt.get('test_split', 'N/A')}", styles['Normal']))
                    story.append(Spacer(1, 6))

                    if "metrics" in mt:
                        data = [["Metric", "Value"]] + [[k, str(v)] for k, v in mt["metrics"].items()]
                        table = Table(data, hAlign="LEFT")
                        table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ]))
                        story.append(table)
                        story.append(Spacer(1, 12))
                else:
                    # If just a string
                    story.append(Paragraph(f"• {mt}", styles['Normal']))
                    story.append(Spacer(1, 6))

        # Footer
        story.append(Spacer(1, 40))
        story.append(Paragraph("☠ DataForge by Mysterious | Hassan", styles['Normal']))

        # Build PDF
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        doc.build(story)
        return pdf_path

    def create_full_report(self, df_clean, activity_log, metrics):
        os.makedirs(self.reports_dir, exist_ok=True)

        # Save cleaned dataset
        clean_path = os.path.join(self.reports_dir, "cleaned_dataset.csv")
        df_clean.to_csv(clean_path, index=False)

        # Save JSON activity log
        json_path = os.path.join(self.reports_dir, "activity_log.json")
        with open(json_path, "w") as f:
            json.dump(activity_log, f, indent=4)

        # Generate PDF report
        pdf_path = self._generate_pdf_report(df_clean, activity_log, metrics)

        # Create ZIP file
        zip_path = os.path.join(self.reports_dir, "dataforge_report.zip")
        with zipfile.ZipFile(zip_path, "w") as zipf:
            zipf.write(clean_path, "cleaned_dataset.csv")
            zipf.write(json_path, "activity_log.json")
            zipf.write(pdf_path, "report.pdf")

            for model_file in os.listdir(self.models_dir):
                zipf.write(os.path.join(self.models_dir, model_file), f"models/{model_file}")

        return zip_path
