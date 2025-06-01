#----------------Importing Required Libraries--------------------
import cv2               # OpenCV for image and video processing
import numpy as np       # NumPy for numerical operations
import pygame            # Pygame for sound playback
import smtplib           # To send emails using SMTP
import datetime          # For timestamp generation
import os                # OS operations (like file handling)
from email.mime.text import MIMEText                       # For plain text email content
from email.mime.multipart import MIMEMultipart             # For multipart email (text + attachments)
from email.mime.base import MIMEBase                       # For attachments
from email import encoders                                 # To encode attachments for email
from fpdf import FPDF      # For generating PDF reports
from persondetection import DetectorAPI                    # Custom person detection class (from TensorFlow API)
import tkinter as tk       # GUI library
from tkinter import messagebox, filedialog, simpledialog   # Extra GUI utilities
import matplotlib.pyplot as plt     # For plotting detection graph
import geocoder           # To get current location via IP

# Initialize sound system
pygame.mixer.init()  # Initialize the Pygame mixer
alert_sound = "alert.mp3" # Path to alert sound file

# Default valariables for crowd detection
CROWD_THRESHOLD = 25            # Default crowd limit
sound_alert_enabled = True      # Flag to control sound alerts
stop_detection = False          # Control to stop detection loop
alert_triggered = False         # To prevent repeated alert sound

# Email Configuration
EMAIL_ADDRESS = "patthembharath32@gmail.com"      # Replace with your sender email
EMAIL_PASSWORD = "vivlwtmxlrwxsyly"        # Replace with your email app password
contact_email_file = "contact_email.txt"      # File to store the contact email

# PDF Report Variables
max_human_count = 0
detection_accuracies = []
detection_source = ""

# -------------------- Email Handling Functions -------------------- #

def get_contact_email():
    if os.path.exists(contact_email_file):
        with open(contact_email_file, "r") as file:
            email = file.read().strip()
            if email:
                return email
    return None

def save_contact_email(email):
    with open(contact_email_file, "w") as file:
        file.write(email)

def get_location():
    try:
        g = geocoder.ip('me')  # Fetches location based on public IP
        if g.latlng:
            latitude, longitude = g.latlng
            return f"https://www.google.com/maps?q={latitude},{longitude}"
    except Exception as e:
        print(f"Failed to get location: {e}")
    return "Location not available"

def send_email_alert():
    receiver_email = get_contact_email()
    if not receiver_email:
        messagebox.showerror("Error", "No contact email found!")
        return
    try:
        location_link = get_location()

        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = receiver_email
        msg['Subject'] = "⚠ Crowd Detection Help Request ⚠"

        body = (f"Hello,\n\nThis is an automated help request from the Crowd Detection System.\n"
                f"🚨 Current Location: {location_link} 🚨\n\n"
                "Please respond promptly.\n\nThank you.")
        msg.attach(MIMEText(body, 'plain'))

        report_filename = "Crowd_Detection_Report.pdf"
        if os.path.exists(report_filename):
            with open(report_filename, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f"attachment; filename={report_filename}")
                msg.attach(part)

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, receiver_email, msg.as_string())
        server.quit()

        messagebox.showinfo("Email Sent", f"Help email sent to {receiver_email} with location!")
    except Exception as e:
        messagebox.showerror("Failed", f"Failed to send email: {e}")

# -------------------- Sound Alert Control Functions-------------------- #

def play_alert():
    global alert_triggered
    if sound_alert_enabled and not alert_triggered:
        pygame.mixer.music.load(alert_sound)
        pygame.mixer.music.play()
        alert_triggered = True

def stop_alert():
    pygame.mixer.music.stop()
    global alert_triggered
    alert_triggered = False

def toggle_sound_alert():
    global sound_alert_enabled
    sound_alert_enabled = not sound_alert_enabled
    status = "Enabled" if sound_alert_enabled else "Disabled"
    messagebox.showinfo("Sound Alert", f"Sound alert {status}")

def update_threshold_dialog():
    global CROWD_THRESHOLD
    new_threshold = simpledialog.askinteger("Update Threshold", "Enter new crowd threshold:")
    if new_threshold is not None:
        CROWD_THRESHOLD = new_threshold
        messagebox.showinfo("Threshold Updated", f"Crowd threshold set to {CROWD_THRESHOLD}.")

def generate_pdf_report():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    avg_accuracy = sum(detection_accuracies) / len(detection_accuracies) if detection_accuracies else 0
    crowd_status = "Overcrowded" if max_human_count > CROWD_THRESHOLD else "Normal"

    # --- Add Report Text ---
    pdf.cell(200, 10, "Crowd Detection Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, f"Detection Source: {detection_source}", ln=True)
    pdf.cell(200, 10, f"Timestamp: {timestamp}", ln=True)
    pdf.cell(200, 10, f"Max Human Count: {max_human_count}", ln=True)
    pdf.cell(200, 10, f"Average Detection Accuracy: {avg_accuracy:.2f}", ln=True)
    pdf.cell(200, 10, f"Crowd Status: {crowd_status}", ln=True)

    # --- Generate Graph ---
    if detection_accuracies:
        plt.figure(figsize=(6, 4))
        plt.plot(range(len(detection_accuracies)), detection_accuracies, marker='o', linestyle='-', color='b', label='Detection Accuracy')
        plt.axhline(y=avg_accuracy, color='r', linestyle='--', label=f'Avg Accuracy: {avg_accuracy:.2f}')
        plt.xlabel("Detection Events")
        plt.ylabel("Accuracy")
        plt.title("Crowd Detection Accuracy Over Time")
        plt.legend()
        plt.grid(True)
        
        # Save the graph as an image
        graph_filename = "crowd_trend.png"
        plt.savefig(graph_filename, dpi=100)
        plt.close()

        # --- Embed Graph in PDF ---
        pdf.image(graph_filename, x=30, y=pdf.get_y() + 10, w=150)  # Adjust position and width

    # Save PDF
    pdf.output("Crowd_Detection_Report.pdf")
    messagebox.showinfo("Report Generated", "Crowd detection report has been saved with a graph.")

def detect_people(video_source, source_name):
    global max_human_count, detection_accuracies, detection_source, stop_detection, alert_triggered
    stop_detection = False
    alert_triggered = False
    detection_source = source_name
    max_human_count = 0
    detection_accuracies.clear()

    video = cv2.VideoCapture(video_source)
    odapi = DetectorAPI()
    threshold = 0.7

    if not video.isOpened():
        messagebox.showerror("Error", f"Unable to open {source_name}!")
        return

    while True:
        check, frame = video.read()
        if not check or stop_detection:
            break

        img = cv2.resize(frame, (1250, 700))
        boxes, scores, classes, num = odapi.processFrame(img)
        person_count = sum(1 for i in range(len(boxes)) if classes[i] == 1 and scores[i] > threshold)

        max_human_count = max(max_human_count, person_count)
        detection_accuracies.extend([scores[i] for i in range(len(scores)) if classes[i] == 1 and scores[i] > threshold])

        for i in range(len(boxes)):
            if classes[i] == 1 and scores[i] > threshold:
                box = boxes[i]
                cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)

        cv2.putText(img, f"Count: {person_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if person_count > CROWD_THRESHOLD:
            play_alert()
            cv2.putText(img, "⚠ CROWD ALERT! ⚠", (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        cv2.imshow(f"Human Detection - {source_name}", img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or cv2.getWindowProperty(f"Human Detection - {source_name}", cv2.WND_PROP_VISIBLE) < 1:
            break

    video.release()
    stop_alert()
    cv2.destroyAllWindows()
    generate_pdf_report()

def select_video():
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", ".mp4;.avi;*.mov")])
    if video_path:
        detect_people(video_path, "Video")

# -------------------- UI with Multiple Frames -------------------- #

def show_frame(frame):
    for f in (start_frame, email_frame, main_frame):
        f.pack_forget()
    frame.pack(fill="both", expand=True)

def start_app():
    if get_contact_email():
        show_frame(main_frame)
    else:
        show_frame(email_frame)

def submit_email():
    email = email_entry.get().strip()
    if email:
        save_contact_email(email)
        messagebox.showinfo("Saved", f"Contact email saved: {email}")
        show_frame(main_frame)
    else:
        messagebox.showerror("Error", "Please enter a valid email.")

root = tk.Tk()
root.title("Real-Time Crowd Detection System")
root.geometry("800x700")
root.configure(bg="#000000")

# -------------------- Frame 1: Start Screen -------------------- #
start_frame = tk.Frame(root, bg="#000000")
start_label = tk.Label(start_frame, text="Crowd Detection System", font=("Arial", 32, "bold"), fg="#FFD700", bg="#000000")
start_label.pack(pady=50)
start_button = tk.Button(start_frame, text="Start", font=("Arial", 18), bg="#00719c", fg="#FFFFFF", command=start_app)
start_button.pack(pady=20)
start_frame.pack(fill="both", expand=True)

# -------------------- Frame 2: Email Input Screen -------------------- #
email_frame = tk.Frame(root, bg="#000000")
email_label = tk.Label(email_frame, text="Enter Emergency Contact Email", font=("Arial", 24, "bold"), fg="#FFD700", bg="#000000")
email_label.pack(pady=30)
email_entry = tk.Entry(email_frame, font=("Arial", 16), width=30)
email_entry.pack(pady=10)
submit_email_button = tk.Button(email_frame, text="Submit", font=("Arial", 16), bg="#00719c", fg="#FFFFFF", command=submit_email)
submit_email_button.pack(pady=20)

# -------------------- Frame 3: Main Interface -------------------- #
main_frame = tk.Frame(root, bg="#000000", padx=20, pady=20)
main_label = tk.Label(main_frame, text="Crowd Detection System", font=("Arial", 24, "bold"), fg="#FFD700", bg="#000000")
main_label.pack(pady=10)

control_container = tk.Frame(main_frame, bg="#000000", bd=5, relief="solid", highlightbackground="#FFD700", highlightthickness=2, padx=20, pady=20)
control_container.pack(pady=10)

buttons_info = [
    ("Detect from Video", select_video, "#00719c"),   
    ("Detect from Camera", lambda: detect_people(0, "Live Camera"), "#00719c"),
    ("Update Threshold", update_threshold_dialog, "#00719c"),
    ("HELP", send_email_alert, "#FF0000"),   
    ("Toggle Sound Alert", toggle_sound_alert, "#00719c"),
    ("Exit", lambda: [root.quit(), setattr(globals(), 'stop_detection', True)], "#00719c")
]

for text, command, color in buttons_info:
    btn = tk.Button(control_container, text=text, command=command, font=("Arial", 14), bg=color, fg="#FFFFFF")
    btn.pack(fill='x', pady=5)

main_frame.pack_forget()

root.mainloop()