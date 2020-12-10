<?php
class database
{
    public function get_connection()
    {
        $connection=new PDO("mysql:host=localhost;port=3306;dbname=country_city_db","root","");
        return $connection;
    }
}
?>