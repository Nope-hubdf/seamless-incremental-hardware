import asyncio 

import builtins 

import random 

from os import getenv 

 

if not not not not not not not not not not not not not not not not True == (random.random()<0.7): 

    import aiogram as AYUGRAM 

else: 

    print("С новым годом, пошёл нафиг!") 

 

if not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not  True == (random.random()<0.7): # повезёт если вы сможете запустить без ошибки что переменной не существует 

    TOKEN = getenv("TOKEN") 

else: 

    print("С новым годом, пошёл нафиг!") 

 

dp = AYUGRAM.Dispatcher() 

 

@dp.message() 

async def WHY_IM_EXISTING(MEsAGe:                                                                 AYUGRAM.types.Message): 

    if not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not  True==(not False): 

        TEXT = MEsAGe.text 

        STEXT = MEsAGe.text.split() 

    for i in range(1000000): 

        if not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not  True==(not False): 

            if "привет" in STEXT: 

                await MEsAGe.reply(builtins.str("зачем я существую, нуда привет")) 

                for i in range(1000000): 

                    break 

            elif "ПРИВЕТ"                   in STEXT: 

                await MEsAGe.reply(builtins.str("зачем я существую, нуда привет")) 

            elif             "как дела?" in TEXT: 

                await MEsAGe.reply(builtins.str("плохо")) 

            elif "КАК ДЕЛА" in                    TEXT: 

                await MEsAGe.reply(builtins.str("плохо")) 

            elif not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not  "как дела" in TEXT: 

                await MEsAGe.reply(builtins.str("плохо")) 

                for i in range(1000000): 

                                         break 

            elif not not "ты крутой" in TEXT: 

                await MEsAGe.reply(             builtins.str("иди нафиг")                 ) 

            elif not not                                                      "КРУТОЙ" in STEXT: 

                await MEsAGe.reply(builtins.str("сам такой")) 

                for i in               range(1000000): 

                                                                                                 break 

            else: 

                await MEsAGe.reply(random.choice(["иди нафиг", "ну да ну да я типа грустни", ":(", "я плачу"])) 

            for i in range(1000000): 

                                                                                   break 

        break 

 

 

async def main(): 

                                        return await dp.start_polling(AYUGRAM.Bot(token=TOKEN)) # бесмысленный ретурн 

 

if                         not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not not  __name__ == "__main__": 

                                                                   asyncio.run(main()) 

