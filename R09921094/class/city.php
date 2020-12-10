<?php
class city
{
    public $db_connect;
    public $id;
    public $city_name;
    public $population;
    public $country_code;
    public function __construct()
    {
        $db=new database();
        $this->db_connect=$db->get_connection();
    }
    public function get_all_city()
    {
        $sql="SELECT * FROM city ";
        $get_data=$this->db_connect->prepare($sql);
        $get_data->execute();
        $all_city=$get_data->fetchAll(PDO::FETCH_ASSOC);
        return $all_city;
    }
    public function create()
    {
        $sql="INSERT INTO city(city_name,population,country_code) VALUES(:city_name,:population,:country_code)";
        $add_data=$this->db_connect->prepare($sql);
        $add_data->bindParam(":city_name",$this->city_name);
        $add_data->bindParam(":population",$this->population);
        $add_data->bindParam(":country_code",$this->country_code);
        return $add_data->execute();
    }
    public function get_one_city()
    {
        $sql="SELECT * FROM city WHERE id=:id";
        $get_data=$this->db_connect->prepare($sql);
        $get_data->bindParam(":id",$this->id);
        $get_data->execute();
        $one_city=$get_data->fetch(PDO::FETCH_ASSOC);
        $this->city_name=$one_city["city_name"];
        $this->population=$one_city["population"];
        $this->country_code=$one_city["country_code"];
    }
    public function update()
    {
        $sql="UPDATE city SET city_name=:city_name,population=:population,country_code=:country_code WHERE id=:id";
        $update_data=$this->db_connect->prepare($sql);
        $update_data->bindParam(":city_name",$this->city_name);
        $update_data->bindParam(":population",$this->population);
        $update_data->bindParam(":country_code",$this->country_code);
        $update_data->bindParam(":id",$this->id);
        return $update_data->execute();
    }
    public function delete()
    {
        $sql="DELETE FROM city WHERE id=:id";
        $delete_data=$this->db_connect->prepare($sql);
        $delete_data->bindParam(":id",$this->id);
        return $delete_data->execute();
    }
}
?>