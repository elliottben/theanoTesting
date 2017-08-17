#python file for sending an email
import smtplib

def sendEmail(message):
    smtplib.SMTP('smtp.gmail.com', 587)
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("elliottmben@gmail.com", "wholehog")

    server.sendmail("elliottmben@gmail.com", "belliott@usc.edu", message)
    server.quit()