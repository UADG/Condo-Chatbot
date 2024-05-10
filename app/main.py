import tensorflow as tf
from flask import Flask,request,abort
import requests
from app.Config import *
import json
import pandas as pd
from pythainlp.tokenize import word_tokenize
from pythainlp.util import dict_trie
from pythainlp.corpus.common import thai_words
from pythainlp.corpus import thai_stopwords
import numpy as np
import re
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences


#  Load model and Load CSV

model = tf.keras.models.Load_model("/app/my_model")
df = pd.read_csv("/app/data.csv")

# Word Index

STOP_WORD = list(thai_stopwords()) + [" ", "\n"]

FORMAT = r"[\u0E00-\u0E7Fa-zA-Z'0-9]+"

words_dict = set(thai_words()) #returns frozenset()
words_dict.add("ครัว")
words_dict.add("แม่บ้าน")
words_dict.add("สะอาด")
words_dict.add("กฏ")
words_dict.add("สังสรร")
words_dict.add("ซัก")

trie = dict_trie(dict_source=words_dict)

def tokenize(sentence):
  return word_tokenize(sentence, engine="newmm",custom_dict=trie)

def cleaning_stop_word(tk_list):
  return [word for word in tk_list if word not in STOP_WORD]

def cleaning_symbols_emoji(tk_list):
  return [re.findall(FORMAT, text)[0] for text in tk_list if re.findall(FORMAT, text)]

X_tokenized = df['question'].astype(str).apply(tokenize).apply(cleaning_symbols_emoji).apply(cleaning_stop_word)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_tokenized)

text_sequences = tokenizer.texts_to_sequences(X_tokenized)
maxlen = max([len(i) for i in text_sequences])

app=Flask(__name__)

@app.route('/webhook',methods=['POST','GET'])

def webhook():
    if request.method=='POST':
        payload =request.json
        Reply_token=payload['events'][0]['replyToken']
        message=payload['events'][0]['message']['text']

        message = message.lower()

        greetings = ["สวัสดีคับ", "สวัสดีครับ", "สวัสดีคร้าบ", "สวัสดีค้าบ", "สวัสดีค่ะ", "สวัสดี", "ทักครับ", "ทักคับ", "ทักคร้าบ", "ทักค้าบ", "ทักค่ะ", "hello", "yo", "hi", "แฮโล่", "โย่"]
        
        if any(greeting in message for greeting in greetings):
            Reply_text="สวัสดีครับ มีอะไรให้ช่วยไหมครับ"
        else:
            input_df = pd.DataFrame({'question':[message]})
            message_tokenized = input_df['question'].astype(str).apply(tokenize).apply(cleaning_symbols_emoji).apply(cleaning_stop_word)
            message_index = tokenizer.texts_to_sequences(message_tokenized)
            message_pad = pad_sequences(message_index,maxlen=maxlen,padding = "pre")
            logit = model.predict(message_pad)
            
            check = True
            for i in range(maxlen) :
                if message_pad[0][i] > 0 :
                    check = False
                    break
            if(np.max(logit)<0.59 or check):
                Reply_text="ขออภัยครับ ผมไม่เข้าใจคำถาม กรุณาพิมพ์คำถามใหม่อีกครั้งนะครับ"
            else:
                ans_class = np.argmax(logit)
                if(ans_class == 0):
                    Reply_text="คอนโดของเราตั้งอยู่ที่ 2034 ถ. พหลโยธิน แขวงเสนานิคม เขตจตุจักร กรุงเทพมหานคร 10900โดยอยู่ตรงกลางระหว่างBTSเสนานิคมและBTSมหาวิทยาลัยเกษตรศาสตร์\
                                สามารถเดินลงจากสถานีเสนานิคมแล้วเดินย้อนมาทางมอเกษตรราวๆ 30 เมตร โดยสถานที่ใกล้ๆ ก็จะมี มหาวิทยาลัยเกษตรศาสตร์ เมเจอร์รัชโยธิน เซนทรัลลาดพร้าว ยูเนี่ยนมอล์ล\
                                ร้านหมูกะทะย่างเนย ชาบูแลป สุกี้ดารา สุกี้ตี๋น้อยพหลโยธิน 19 สามารถโดยสารโดย BTS สายสีเขียวอ่อน หรือรถเมล์เส้นพหลโยธิน ได้เลย"
                elif(ans_class == 1):
                    Reply_text="> ค่าน้ำ 8 บาท  ต่อหน่วย, \
                                > ค่าไฟ 18 บาท ต่อหน่วย, \
                                > ค่าส่วนกลาง 500 บาท ต่อเดือน, \
                                โดยค่าใช้จ่ายทั้งหมดเท่ากันทุกขนาดห้อง"
                elif(ans_class == 2):
                    Reply_text="> ค่ามัดจำ  5000 บาท, \
                                โดยต้องจ่ายก่อนเข้าอยู่ สามารถจ่ายได้ทั้งการโอนและเงินสดที่หน้าเคาเตอร์\
                                โดยค่ามัดจำจะได้คืน 6 เดือนภายหลังจากการเข้าอยู่\
                                เมื่อถึงกำหนดเวลาเงินจะเข้าบัญชีภายใน 2 สัปดาห์\
                                โดยค่ามัดจำนั้นจะถูกหักเมื่อทำผิดข้อกำหนดของคอนโด โดยจะถูกหักตามความร้ายแรงของการกระทำความผิด"
                elif(ans_class == 3):
                    Reply_text="ห้ามส่งเสียงดังรบกวนผู้อื่น, \
                                สามารถทำอาหารภายในห้องได้ โดยอนุญาติให้ใช้แค่เตาไฟฟ้าที่ทางคอนโดมีให้ แต่ไม่อนุญาติให้ติดตั้งแก๊สหุงต้ม, \
                                ห้ามนำวัตถุไวไฟหรือวัตถุอันตรายเข้ามาภายในคอนโด, \
                                ห้ามสูบบุหรี่ภายในห้อง ทางคอนโดได้เตรียมพื้นที่สูบบุหรี่ไว้แล้ว, \
                                โปรดรักษาความสะอาดภายในคอนโด และทิ้งขยะภายในโซนที่เตรียมไว้ให้ในแต่ละชั้น, \
                                ห้ามเลี้ยงสัตว์ภายในคอนโดทุกชนิด, \
                                ห้ามทำการกระทำที่จะเป็นการรบกวนผู้อื่น, \
                                ห้ามเจาะ รึทำการติดตั้งอะไรเพิ่มเติมภายในห้อง เช่น การเจาะผนัง หรือรีโนเวทพื้นที่ต่างๆ ภายในห้องและนอกห้อง, \
                                ห้ามทำลายทรัพย์สินของคอนโดทุกชนิด, \
                                หากมีการกระทำที่ทำให้เกิดการเสียหาย จะต้องรับผิดชอบไม่ว่ากรณีใด"
                elif(ans_class == 4):
                    Reply_text="คอนโดของเรามีทั้งหมด 3 ขนาด คือ S,M,L, \
                                ห้องไซซ์ S ราคา 5000 บาทต่อเดือน ขนาด 140 ตร.ม., \
                                ห้องไซซ์ M ราคา 7000 บาทต่อเดือน ขนาด 150 ตร.ม., \
                                ห้องไซซ์ L ราคา 8000 บาทต่อเดือน ขนาด 165 ตร.ม., \
                                โดยสามารถติดต่อขอเข้ามาดูห้องตัวอย่างได้ที่คอนโดเราเลยค่ะ"
                elif(ans_class == 5):
                    Reply_text="สามารถชำระเงินได้ทุกวันที่เริ่มต้นสัญญาของทุกเดือน สามารถชำระผ่านทางออนไลน์หรือเงินสดก็ได้ แต่ไม่รับบัตรเครดิต\
                                สามารถจ่ายช้าได้ไม่เกิน 5 วันจากวันที่ต้องชำระเงิน\
                                หากจ่ายช้าเกิน 5 วันจากวันที่ต้องชำระเงิน ต้องเสียค่าปรับ 10% จากค่าห้อง\
                                หากจ่ายช้าเกิน 12 วันจะยึดคียการ์ดไม่ให้เข้าห้อง และต้องเสียค่าปรับ 17% จากค่าห้องและถ้าหากจ่ายช้าเกิน 18 วันจะถูกยึดห้อง"
                elif(ans_class == 6):
                    Reply_text="มีลิฟต์ 2 ตัวภายในคอนโด, \
                                มีเครื่องซักผ้า และอบผ้าหยอดเหรียญบริการอยู่บริเวณส่วนกลาง ด้านล่าง, \
                                มี co-working space ให้ใช้บริการ 24 ชม., \
                                มีสวนหย่อมสามารถเข้าได้ 24 ชม., \
                                ฟิตเนสเปิดให้ใช้บริการในช่วงเวลา 07.00 - 22.00 น. โดยไม่เสียค่าใช้จ่ายในการเข้าใช้, \
                                สระว่ายน้ำมีทั้งของเด็กและผู้ใหญ่โดยที่จะให้บริการในช่วงเวลา 07.00 - 22.00 น. โดยไม่เสียค่าใช้จ่าย, \
                                ที่จอดรถมีพื้นที่ ให้ 1 ห้อง 1 ที่จอด โดยไม่เสียค่าใช้จ่าย โดยหากท่านต้องการต้องการพื้นที่จอดรถมากกว่า 1 ที่, \
                                ต้องติดต่อลงทะเบียนกับเจ้าหน้าที่ โดยจะเสียค่าที่จอดเดือนละ 500 บาท, \
                                มีที่ชาร์จรถ EV ให้บริการ 2 เตรื่อง, \
                                มีแม่บ้านประจำคอนโดทำความสะอาดทุกวัน, \
                                มีที่ทิ้งขยะประจำทุกชั้น, \
                                มีการปรับปรุง ตรวจสอบ บำรุงรักษาคอนโด ทุกๆ 6 เดือน"

                elif(ans_class == 7):
                    Reply_text="ห้องไซซ์ S มีโต๊ะยาว 1 ตัว เก้าอี้ 2 ตัว เตียงเดี่ยว ไมโครเวฟ 1 เครื่อง เตาไฟฟ้า 1 เครื่อง แอร์ 1 เครื่อง,  \
                                ห้องไซซ์ M มีโต๊ะยาว1ตัว เก้าอี้ 2 ตัว เตียงเดี่ยว ไมโครเวฟ 1 เครื่อง เตาไฟฟ้า 1 เครื่อง แอร์ 1 เครื่อง,  \
                                ห้องไซซ์ L มีโต๊ะยาว 2 ตัว เก้าอี้ 4 ตัว เตียงเดี่ยว ไมโครเวฟ 1 เครื่อง เตาไฟฟ้า 1 เครื่อง แอร์ 1 เครื่อง"
                elif(ans_class == 8):
                    Reply_text="มีกล้องวงจรปิดรอบตึก หน้าทางเข้า-ออก\
                                มีกล้องตามแต่ละชั้น มีรปภ. ตลอด 24. ชม.\
                                เวลามีปัญหาสามารถติดต่อ รปภ. ได้ตลอด 24 ชม.\
                                มีระบบ Access Control ด้วย Key Card เข้า-ออกอาคาร และพื้นที่จอดรถ"

        print(Reply_text,flush=True)
        ReplyMessage(Reply_token,Reply_text,Channel_access_token)
        return request.json,200
    elif request.method=='GET':
        return "this is method GET!!!",200
    else:
        abort(400)


def ReplyMessage(Reply_token,TextMessage,Line_Acees_Token):
    LINE_API='https://api.line.me/v2/bot/message/reply/'

    Authorization='Bearer {}'.format(Line_Acees_Token)
    print(Authorization)
    headers={
        'Content-Type':'application/json; char=UTF-8',
        'Authorization':Authorization
    }

    data={
        "replyToken":Reply_token,
        "messages":[{
            "type":"text",
            "text":TextMessage
        }
        ]
    }
    data=json.dumps(data) # ทำเป็น json
    r=requests.post(LINE_API,headers=headers,data=data)
    return 200
