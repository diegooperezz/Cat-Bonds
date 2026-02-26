import numpy as np
import polars as pl
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
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
            
            # Métricas de medición de la tormenta
            storm_ace = storm.ace if hasattr(storm, 'ace') else 0
            storm_max_wind = np.nanmax(storm.vmax) if hasattr(storm, 'vmax') else 0

            storm_data.append({
                "EventName": f"{storm.name}_{year}",
                "start_date": start_date,
                "end_date": end_date,
                "ACE": storm_ace,
                "MaxWind": storm_max_wind
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
df_storms = pl.read_parquet(r"1 Prepara Datos\storms_catalog.parquet")

#_____________________________________________________________________________________________
# 2. -  FILTRAR RECLAMACIONES QUE COINCIDAN CON LAS TORMENTAS CON NOMBRE

# Cargar el dataset como LazyFrame (lectura sin cargar en memoria inmediatamente)
raw_claims = (pl.scan_parquet(r"0 Datos\FimaNfipClaimsV2.parquet")
              .with_row_index(name="ClaimID", offset=0))

# Comprobamos que tenemos la fecha de pérdida en formato datetime y ordenamos por esta fecha
raw_claims = raw_claims.with_columns(
    pl.col("dateOfLoss").cast(pl.Datetime)
)

# Creamos una columna que almacene el coste total de las reclamaciones (Building + Contents)
total_claims = (raw_claims.with_columns([
        pl.col("amountPaidOnBuildingClaim").fill_null(0),
        pl.col("amountPaidOnContentsClaim").fill_null(0)
    ])
    .with_columns(
        (pl.col("amountPaidOnBuildingClaim") + pl.col("amountPaidOnContentsClaim"))
        .alias("TotalLoss")
    )
)

# Preparamos el DataFrame de reclamaciones para hacer un join con el DataFrame de tormentas

current_year = max(df_storms.select(pl.col("end_date").dt.year()).to_series()) + 1 
inf_factor = 1.03  # Factor de inflación anual (3% en este caso)

claims_dated = (total_claims.filter(pl.col("dateOfLoss").is_not_null()).with_columns([
        pl.col("dateOfLoss").cast(pl.Datetime),
        (inf_factor ** (current_year - pl.col("dateOfLoss").dt.year())).alias("InflationFactor")
    ])).sort("dateOfLoss")

# Ajustamos el coste total de las reclamaciones por inflación para que sean comparables a lo largo del tiempo

claims_dated = claims_dated.with_columns(
    (pl.col("TotalLoss") * pl.col("InflationFactor")).alias("AdjustedTotalLoss")
)

# Realizamos un join entre las reclamaciones y las tormentas para encontrar coincidencias

df_storms = df_storms.with_columns(
    pl.col("start_date").dt.year().alias("join_year")
)

df_claims_prep = claims_dated.with_columns(
    pl.col("dateOfLoss").dt.year().alias("join_year")
)

df_linked = (
    df_claims_prep.join(
        df_storms.lazy(),
        on="join_year",  # Unimos solo siniestros 2005 con tormentas 2005
        how="inner"
    )
    # 2. Ahora aplicamos la lógica real de intervalo
    .filter(
        (pl.col("dateOfLoss") >= pl.col("start_date")) & 
        (pl.col("dateOfLoss") <= pl.col("end_date")) &
        (pl.col("TotalLoss") > 0)
    )
    # Ordenamos por identificador y magnitudes de intensidad de la tormenta
    .sort(["ClaimID", "MaxWind", "ACE"], descending=[False, True, True])
    # Nos quedamos solo con la primera coincidencia por siniestro (la correspondiente a la tormenta más intensa)
    .unique(subset="ClaimID", keep="first")   
    # Nos quedamos solo con las columnas relevantes para el análisis posterior
    .select([
        "ClaimID",
        "dateOfLoss", 
        "state",  
        "EventName", 
        "ACE",
        "MaxWind",
        "TotalLoss",
        "AdjustedTotalLoss"
    ])
    .collect()
)


#_________________________________________________________________________________________
# 3. -  BREVE ANÁLISIS EXPLORATORIO DE LOS DATOS VINCULADOS

# Vemos cuales son las tormentas con mayor número de reclamaciones vinculadas y mayor coste total y graficamos los resultados

top_storms = (df_linked.group_by("EventName")
    .agg([
        pl.count().alias("NumClaims"),
        pl.sum("AdjustedTotalLoss").alias("AdjustedTotalLossByStorm"),
        pl.sum("TotalLoss").alias("TotalLossByStorm")
    ])
    .sort("AdjustedTotalLossByStorm", descending=True)
)

# Gráfico de barras para las 30 tormentas con mayor coste total ajustado de reclamaciones
plt.figure(figsize=(12, 6))
sns.barplot(data=top_storms.head(30).to_pandas(), x="EventName", y="AdjustedTotalLossByStorm")
plt.xticks(rotation=90)
plt.title("Top 30 Tormentas por Coste Total Ajustado de Reclamaciones")
plt.xlabel("Evento")   
plt.ylabel("Coste Total Ajustado")
plt.tight_layout()
plt.show()

# Gráfico de barras para las 30 tormentas con mayor número de reclamaciones vinculadas
plt.figure(figsize=(12, 6)) 
sns.barplot(data=top_storms.head(30).to_pandas(), x="EventName", y="NumClaims")
plt.xticks(rotation=90)
plt.title("Top 30 Tormentas por Número de Reclamaciones Vinculadas")
plt.xlabel("Evento")
plt.ylabel("Número de Reclamaciones")
plt.tight_layout()
plt.show()

# Gráfico de dispersión para analizar la relación entre el número de reclamaciones y el coste total ajustado por tormenta
plt.figure(figsize=(10, 6))
sns.scatterplot(data=top_storms.to_pandas(), x="NumClaims", y="AdjustedTotalLossByStorm")
plt.title("Relación entre Número de Reclamaciones y Coste Total Ajustado por Tormenta")
plt.xlabel("Número de Reclamaciones")
plt.ylabel("Coste Total Ajustado")
plt.tight_layout()
plt.show()

# Por otro lado, analizamos la serie temporal de reclamaciones vinculadas a tormentas a lo largo del tiempo para ver si hay alguna tendencia o patrón estacional

# Granularidad anual
claims_over_time = (df_linked.group_by(pl.col("dateOfLoss").dt.year().alias("Year"))
    .agg([pl.sum("AdjustedTotalLoss").alias("AdjustedTotalLossByYear"),
          pl.sum("TotalLoss").alias("TotalLossByYear"),
          pl.count().alias("NumClaimsByYear")])
    .sort("Year")
)

# Gráfico de líneas para el coste total ajustado de reclamaciones vinculadas a tormentas a lo largo del tiempo
plt.figure(figsize=(10, 6))
sns.lineplot(data=claims_over_time.to_pandas(), x="Year", y="AdjustedTotalLossByYear")
# Añadimos el coste total sin ajustar para comparar
sns.lineplot(data=claims_over_time.to_pandas(), x="Year", y="TotalLossByYear", label="Total Loss Sin Ajustar")
plt.title("Coste Total Ajustado de Reclamaciones Vinculadas a Tormentas a lo largo del Tiempo")
plt.xlabel("Año")
plt.ylabel("Coste Total Ajustado")
plt.tight_layout()
plt.show()

# Gráfico de líneas para el número de reclamaciones vinculadas a tormentas a lo largo del tiempo
plt.figure(figsize=(10, 6))
sns.lineplot(data=claims_over_time.to_pandas(), x="Year", y="NumClaimsByYear")
plt.title("Número de Reclamaciones Vinculadas a Tormentas a lo largo del Tiempo")
plt.xlabel("Año")
plt.ylabel("Número de Reclamaciones")
plt.tight_layout()
plt.show()


# Granularidad mensual
claims_over_time_monthly = (
    df_linked
    .group_by(
        # Esto convierte "2023-08-23" en "2023-08-01"
        pl.col("dateOfLoss").dt.truncate("1mo").alias("YearMonth") 
    )
    .agg([
        pl.sum("AdjustedTotalLoss").alias("AdjustedTotalLossByMonth"),
        pl.sum("TotalLoss").alias("TotalLossByMonth"),
        pl.len().alias("NumClaimsByMonth") # Nota: pl.len() es más rápido que count()
    ])
    .sort("YearMonth")
)

# Gráfico de líneas para el coste total ajustado de reclamaciones vinculadas a tormentas a lo largo del tiempo (granularidad mensual)
plt.figure(figsize=(12, 6))
sns.lineplot(data=claims_over_time_monthly.to_pandas(), x="YearMonth", y="AdjustedTotalLossByMonth")
# Añadimos el coste total sin ajustar para comparar
sns.lineplot(data=claims_over_time_monthly.to_pandas(), x="YearMonth", y="TotalLossByMonth", label="Total Loss Sin Ajustar")
plt.title("Coste Total Ajustado de Reclamaciones Vinculadas a Tormentas a lo largo del Tiempo (Mensual)")
plt.xlabel("Año-Mes")
plt.ylabel("Coste Total Ajustado")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Gráfico de Severidad + Frecuencia para analizar la relación entre el número de reclamaciones y el coste total ajustado por mes
pdf = claims_over_time_monthly.to_pandas()

fig, ax1 = plt.subplots(figsize=(15, 7))

# --- Eje Izquierdo: Coste (Barras o Área) ---
color = 'tab:blue'
ax1.set_xlabel('Fecha del Siniestro')
ax1.set_ylabel('Coste Total Ajustado (USD)', color=color)
ax1.bar(pdf['YearMonth'], pdf['AdjustedTotalLossByMonth'], width=20, color=color, alpha=0.6, label='Coste Total')
ax1.tick_params(axis='y', labelcolor=color)

# Formato de moneda en eje Y
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# --- Eje Derecho: Número de Reclamaciones (Línea) ---
ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Número de Reclamaciones', color=color)  
ax2.plot(pdf['YearMonth'], pdf['NumClaimsByMonth'], color=color, linewidth=2, label='Nº Reclamaciones')
ax2.tick_params(axis='y', labelcolor=color)

# --- Mejoras visuales ---
plt.title('Evolución Mensual: Coste vs. Frecuencia de Siniestros (2000-2024)')
ax1.xaxis.set_major_locator(mdates.YearLocator(2)) # Mostrar año cada 2 años
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.tight_layout()  

plt.show()


###### Caída de la severidad desde unos años a esta parte debido a que los siniestros si bien han sido reportados, aún no han sido cerrdos ni pagados en su totalidad, hay que buscar un factor de corrección
# para ajustar el coste total de las reclamaciones más recientes y que aún no han sido cerradas (Buscar IBNR puro y de desarrollo y RBNS)
# Aunque número de reclamaciones también baja, comprobar si son necesarios estos ajustes.

