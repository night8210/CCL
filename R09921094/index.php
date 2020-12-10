<!DOCTYPE html>
<html lang="en">
<head>
    <?php include "./layout/header.php";?>
</head>
<body>
    <?php include "./layout/navbar.php";?>
    <div class="container my-5">
        <div class="row">
            <div class="col">
                <table id="city-table" class="table table-striped table-bordered table-hover" width="100%">
                    <thead>
                        <tr>
                            <th scope="col">ID</th>
                            <th scope="col">城市名字</th>
                            <th scope="col">城市人口</th>
                            <th scope="col">國家代碼</th>
                            <th scope="col">編輯/刪除</th>
                        </tr>
                    </thead>
                    <tbody>
                        <?php
                        include "./class/database.php";
                        include "./class/city.php";
                        $city=new city();
                        $data=$city->get_all_city();
                        ?>
                        <?php foreach($data as $row):?>
                        <tr>
                            <td><?=$row["id"]?></td>
                            <td><?=$row["city_name"]?></td>
                            <td><?=$row["population"]?></td>
                            <td><?=$row["country_code"]?></td>
                            <td>
                                <a href='./edit.php?id=<?=$row["id"]?>' class="btn btn-outline-info">
                                    <i class="fas fa-edit"></i>
                                </a>
                                <button class="btn btn-outline-secondary delete-btn" delete-id="<?=$row["id"]?>">
                                    <i class="fas fa-trash-alt"></i>
                                </button>
                            </td>
                        </tr>
                        <?php endforeach;?>
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    <?php include "./layout/footer.php";?>
    <script>
        $(document).ready(function()
        {
            $('#city-table').DataTable();
            $('#city-table').on('click','.delete-btn',function()
            {
                $id=$(this).attr('delete-id');
                var confirm_result=confirm('確定是否刪除');
                if(confirm_result)
                {
                    $.ajax(
                    {
                        url:'./delete.php',
                        type:'GET',
                        data:
                        {
                            id:$id
                        },
                        success:function(data)
                        {
                            if(data==1)
                            {
                                alert('刪除成功');
                                location.reload();
                            }
                            else
                            {
                                alert('刪除失敗，稍後再試');
                            }
                        }
                    })
                }
            })
        });
    </script>
</body>
</html>