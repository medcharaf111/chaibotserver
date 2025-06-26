import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr

GMAIL_USER = 'tutifytest@gmail.com'
GMAIL_PASS = 'bkpiqehtbfbpkajx'

recipient = 'charafamri1111@gmail.com'
subject = 'SMTP Test from FaceRec App'
body = 'This is a test email sent from the FaceRec App SMTP setup.'

msg = MIMEText(body, 'plain')
msg['From'] = formataddr(("FaceRec App", GMAIL_USER))
msg['To'] = recipient
msg['Subject'] = subject

try:
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(GMAIL_USER, GMAIL_PASS)
        server.send_message(msg)
    print(f"Test email sent to {recipient}!")
except Exception as e:
    print(f"Failed to send email: {e}") 