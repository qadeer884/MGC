-- Query 1: Conversion rate by source (only sources with 200+ leads)
SELECT
    source,
    COUNT(*) AS total_leads,
    SUM(converted) AS converted_leads,
    ROUND(100.0 * SUM(converted) / COUNT(*), 2) AS conversion_rate
FROM leads
GROUP BY source
HAVING COUNT(*) >= 200
ORDER BY conversion_rate DESC;

-- Query 2: Find duplicates (based on hash)
SELECT
    crm_record_hash,
    COUNT(*) AS duplicate_count,
    GROUP_CONCAT(lead_id) AS lead_ids  
FROM leads
GROUP BY crm_record_hash
HAVING COUNT(*) > 1;