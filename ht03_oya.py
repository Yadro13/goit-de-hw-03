from argparse import ArgumentParser
from pathlib import Path

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DecimalType


def main():
    parser = ArgumentParser(description="Витрати за категоріями товарів")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    data_dir = parser.parse_args().data_dir.resolve()

    spark = (
        SparkSession.builder.appName("GoIT Data Engineering HW 03")
        .master("local[2]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    def read_csv(name, schema):
        return spark.read.csv(
            str(data_dir / f"{name}.csv"),
            header=True,
            schema=schema,
            enforceSchema=False,
            ignoreLeadingWhiteSpace=True,
            ignoreTrailingWhiteSpace=True,
        )

    try:
        # 1. Кожен CSV читаємо в окремий DataFrame.
        users = read_csv("users", "user_id INT, name STRING, age INT, email STRING")
        purchases = read_csv(
            "purchases", "purchase_id INT, user_id INT, product_id INT, date DATE, quantity INT"
        )
        products = read_csv(
            "products", "product_id INT, product_name STRING, category STRING, price DECIMAL(12,2)"
        )

        print("=== p1 | Вхідні DataFrame ===", flush=True)
        print("users.csv (вік, вибірка без персональних даних)")
        users.orderBy("user_id").select("age").show(5)
        print("purchases.csv")
        purchases.orderBy("purchase_id").select("date", "quantity").show(5)
        print("products.csv")
        products.orderBy("product_id").select("category", "price").show(5)

        # 2. Видаляємо рядки з пропусками в будь-якому стовпці.
        users_clean = users.na.drop(how="any")
        purchases_clean = purchases.na.drop(how="any")
        products_clean = products.na.drop(how="any")
        print("=== p2 | Очищення даних ===", flush=True)
        row_counts = []
        for name, raw, clean in [
            ("users.csv", users, users_clean),
            ("purchases.csv", purchases, purchases_clean),
            ("products.csv", products, products_clean),
        ]:
            before, after = raw.count(), clean.count()
            row_counts.append((name, before, after, before - after))
        spark.createDataFrame(row_counts, ["file", "before", "after", "removed"]).show()

        purchases_with_category = (
            purchases_clean.join(products_clean.select("product_id", "category", "price"), "product_id")
            .withColumn("amount", F.col("quantity") * F.col("price"))
        )

        # 3. Сума покупок за категоріями для всіх покупців.
        category_totals = purchases_with_category.groupBy("category").agg(
            F.sum("amount").alias("total_spend")
        )
        print("=== p3 | Загальні витрати за категоріями ===", flush=True)
        category_totals.orderBy(F.desc("total_spend"), "category").show()

        # 4. Сума покупок за категоріями для користувачів віком 18–25 включно.
        users_18_25 = users_clean.where(F.col("age").between(18, 25)).select("user_id")
        category_totals_18_25 = (
            purchases_with_category.join(users_18_25, "user_id")
            .groupBy("category")
            .agg(F.sum("amount").alias("spend_18_25"))
        )
        print("=== p4 | Витрати за категоріями, 18–25 років ===", flush=True)
        category_totals_18_25.orderBy(F.desc("spend_18_25"), "category").show()

        # 5. Частка категорії від усіх витрат вікової групи.
        cohort_total = category_totals_18_25.agg(F.sum("spend_18_25").alias("cohort_total"))
        category_shares = (
            category_totals_18_25.crossJoin(cohort_total)
            .withColumn("share_pct", F.try_divide(F.col("spend_18_25") * 100, F.col("cohort_total")))
        )
        ranked_shares = category_shares.orderBy(F.desc_nulls_last("share_pct"), "category")
        display_columns = [
            "category",
            "spend_18_25",
            F.round("share_pct", 2).cast(DecimalType(10, 2)).alias("share_pct"),
        ]
        print("=== p5 | Частка витрат за категоріями, % ===", flush=True)
        ranked_shares.select(*display_columns).show()

        # 6. Ранжуємо за часткою до округлення. Категорія визначає порядок за рівності.
        print("=== p6 | Три категорії з найбільшою часткою ===", flush=True)
        ranked_shares.where(F.col("share_pct").isNotNull()).limit(3).select(*display_columns).show()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
