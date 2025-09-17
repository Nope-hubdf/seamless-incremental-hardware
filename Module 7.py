#1

# seasons = ("Winter", "Spring", "Summer", "Autumn")
# month = int(input("Give month number (1-12): "))
# if month == 12 or month == 1 or month == 2:
#     print("Season is", seasons[0])
# elif month == 3 or month == 4 or month == 5:
#     print("Season is", seasons[1])
# elif month == 6 or month == 7 or month == 8:
#     print("Season is", seasons[2])
# elif month == 9 or month == 10 or month == 11:
#     print("Season is", seasons[3])
# else:
#     print("That is not a real month")

#2

# names = set()
# for i in range(5):
#     name = input("Enter a name: ")
#     if name == "":
#         break
#     if name in names:
#         print("This is an existing name.")
#     else:
#         print("This is a new name.")
#         names.add(name)
# print("These are the names: ")
# for n in names:
#     print(n)


#3
import mysql.connector

connection = mysql.connector.connect(
host='127.0.0.1',
          port= 3306,
          database='airport',
          user='manik',
          password='0000',
          autocommit=True
         )
airports = {}
print("1 = Add Airport")
print("2 = Fetch Airport")
print("3 = Quit")
for i in range(10):
    if i == 0:
        option = input("Enter: ")
    else:
        option = input("Enter again: ")
    if option == "1":
        code = input("Enter ICAO code: ")
        name = input("Enter Airport name: ")
        airports[code] = name
    elif option == "2":
        code = input("Enter ICAO code: ")
        if code in airports:
            print(airports[code])
        else:
            print("Not a valid ICAO code.")
    elif option == "3":
        break
    else:
        print("Not a valid option.")
print("Program ended.")

