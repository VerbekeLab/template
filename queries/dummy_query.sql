-- Dummy query to select and aggregate sales data

SELECT
    product_id,
    product_name,
    category,
    SUM(quantity_sold) AS total_quantity_sold,
    SUM(total_price) AS total_revenue,
    AVG(total_price) AS average_price,
    COUNT(*) AS total_transactions
FROM
    sales
WHERE
    sale_date BETWEEN '2023-01-01' AND '2023-12-31'
GROUP BY
    product_id,
    product_name,
    category
ORDER BY
    total_revenue DESC;
