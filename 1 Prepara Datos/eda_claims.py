import polars as pl
from datetime import timedelta
import tropycal.tracks as tracks

#____________________________________________________________________________________________
# 1. - GENERAR DATAFRAME DE LAS TORMENTAS DE RENOMBRE 

basin = tracks.TrackDataset(basin='north_atlantic', source='ibtracs', include_btk=False)

start_year = 2000
end_year = 2024

# Añadimos un buffer de días antes y después de cada tormenta para capturar posibles reclamaciones relacionadas
days_before = 3 
days_after = 5

# Lista para guardar los datos antes de pasar a Polars
storm_data = []

for year in range(start_year, end_year + 1):
    try:
        storm_ids = basin.filter_storms(year_range = (year,year))
        for sid in storm_ids:
            storm = basin.get_storm(sid)
            if storm.name == 'UNNAMED': continue
            
            # Extraemos fechas y añadimos buffer
            start_date = min(storm.time) - timedelta(days=days_before)
            end_date = max(storm.time) + timedelta(days=days_after)
            
            storm_data.append({
                "EventName": f"{storm.name}_{year}",
                "start_date": start_date,
                "end_date": end_date
            })
    except:
        continue

# Convertimos la lista de diccionarios a un DataFrame de Polars y ordenamos por fecha de inicio
df_storms = pl.DataFrame(storm_data)

df_storms = df_storms.with_columns([
    pl.col("start_date").cast(pl.Datetime),
    pl.col("end_date").cast(pl.Datetime)
]).sort("start_date")

#  Guardamos el DataFrame de tormentas para su uso posterior
df_storms.write_parquet(r"1 Prepara Datos\storms_catalog.parquet")


#_____________________________________________________________________________________________
# 2. -  FILTRAR RECLAMACIONES QUE COINCIDAN CON LAS TORMENTAS CON NOMBRE

# Cargar el dataset como LazyFrame (lectura sin cargar en memoria inmediatamente)
total_claims = pl.scan_parquet(r"0 Datos\FimaNfipClaimsV2.parquet")

# Comprobamos que tenemos la fecha de pérdida en formato datetime y ordenamos por esta fecha
total_claims = total_claims.with_columns(
    pl.col("dateOfLoss").cast(pl.Datetime).sort("dateOfLoss")
)