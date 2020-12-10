<?php
class country
{
    public $db_connect;
    public function __construct()
    {
        $db=new database();
        $this->db_connect=$db->get_connection();
    }    
    public function get_all_country_code()
    {
        $sql="SELECT country_code FROM country";
        $get_data=$this->db_connect->prepare($sql);
        $get_data->execute();
        $all_country_code=$get_data->fetchAll(PDO::FETCH_ASSOC);
        return $all_country_code;
    }
}
?>