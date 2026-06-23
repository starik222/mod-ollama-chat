INSERT INTO mod_ollama_chat_personality (guid, personality)
SELECT 
    c.guid,
    (
        SELECT `key` 
        FROM mod_ollama_chat_personality_templates 
        -- Exclude templates meant only for manual assignment
        WHERE manual_only = 0 
        -- This dummy condition forces MySQL to evaluate the subquery per row
        AND c.guid IS NOT NULL 
        ORDER BY RAND() 
        LIMIT 1
    ) AS random_personality
FROM characters c
LEFT JOIN mod_ollama_chat_personality p 
    ON c.guid = p.guid
WHERE p.guid IS NULL;