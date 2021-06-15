# -*- coding: utf-8 -*-
"""
Created on Mon May  3 11:40:55 2021

@author: limon
"""


import pika, sys, os
from playsound import playsound

def main():
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()

    channel.queue_declare(queue='hello')

    def callback(ch, method, properties, body):
        
        try:
            playsound('alarm.wav')
            
        except:
            pass
        
        
        print("Received Message")

    channel.basic_consume(queue='hello', on_message_callback=callback, auto_ack=True)

    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()
    

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)