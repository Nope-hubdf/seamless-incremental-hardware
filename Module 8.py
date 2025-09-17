import mysql.connector
from geopy.distance import geodesic

connection = mysql.connector.connect(
host='127.0.0.1',
          port= 3306,
          database='airport',
          user='manik',
          password='0000',
          autocommit=True
         )
cursor = connection.cursor()

#1
# icao = input("Please enter an ICAO code: ").upper().strip()
# sql = "SELECT name, municipality FROM airport WHERE ident = %s"
# cursor.execute(sql, (icao,))
# row = cursor.fetchone()
# if row:
#     print("\nAirport found!")
#     print("Name:", row[0])
#     print("Town:", row[1])
# else:
#     print("\nNot a valid ICAO code.")

#2
# country = input("Enter area code: ")
# country = country.upper().strip()
# sql = """
#     SELECT type, COUNT(*)
#     FROM airport
#     WHERE iso_country = %s
#     GROUP BY type
#     ORDER BY type;
# """
# cursor.execute(sql, (country,))
# rows = cursor.fetchall()
# if rows:
#     print(f"\nAirports in {country}:")
#     for row in rows:
#         airport_type = row[0]
#         count = row[1]
#         print(f"- {count} {airport_type}(s)")
# else:
#     print("\nNot a valid area code.")

#3
icao1 = input("Enter ICAO code of an airport: ").upper()
icao2 = input("Enter ICAO code of another airport: ").upper()
def get_airport_info(icao_code):
    query = "SELECT name, latitude_deg, longitude_deg FROM airport WHERE ident = %s"
    cursor.execute(query, (icao_code,))
    result = cursor.fetchone()
    if result:
        name, lat, lon = result
        return {"name": name, "coords": (lat, lon)}
    else:
        return None
airport1 = get_airport_info(icao1)
airport2 = get_airport_info(icao2)
if airport1 and airport2:
    distance = geodesic(airport1["coords"], airport2["coords"]).kilometers
    print(f"The distance between {airport1['name']} and {airport2['name']} is approximately {distance:.2f} km.")
else:
    print("Error! Either one or both of the ICAO codes you entered were not found in the database.")

cursor.close()
connection.close()
