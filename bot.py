import telebot
from tensorflow import keras
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TOKEN = "682921197:AAFyc3Y_184UEU_tpJX6XMinyhlWVfJu1Uo"

bot = telebot.TeleBot(TOKEN)

model_covid = keras.models.load_model('x_ray_COVID19.h5')
model_pnevm = keras.models.load_model('x_ray_PNEUMONIA.h5')

@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, "Добро пожаловать в лучшего бота!".format(message.from_user, bot.get_me()), parse_mode='html')

@bot.message_handler(content_types=['photo'])
def handle_docs_photo(message):
    try:
        file_info = bot.get_file(message.photo[len(message.photo)-1].file_id)
        
        downloaded_file = bot.download_file(file_info.file_path)
        
        src = 'tmp/' + file_info.file_path
        
        with open(src, 'wb') as new_file:
           new_file.write(downloaded_file)
           
        bot.reply_to(message,"Фото добавлено")

        img = load_img(src, target_size=(150, 150))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        features_covid = model_covid.predict(x)
        features_pnevm = model_pnevm.predict(x)
        if (features_covid == 1.0):
            bot.send_message(message.chat.id, "Наша нейронная сеть вывела у вас COVID-19. Вам стоит обратиться к врачу!".format(message.from_user, bot.get_me()), parse_mode='html')
        else:
            bot.send_message(message.chat.id, "Наша нейронная сеть не вывела у вас COVID-19".format(message.from_user, bot.get_me()), parse_mode='html')
        if (features_pnevm == 1.0):
            bot.send_message(message.chat.id, "Наша нейронная сеть вывела у вас пневмонию. Вам стоит обратиться к врачу!".format(message.from_user, bot.get_me()), parse_mode='html')
        else:
            bot.send_message(message.chat.id, "Наша нейронная сеть не вывела у вас пневмонии".format(message.from_user, bot.get_me()), parse_mode='html')

        os.remove(src)
    except Exception as e:
        bot.reply_to(message,e )

if __name__ == '__main__':
    bot.polling(none_stop=True)