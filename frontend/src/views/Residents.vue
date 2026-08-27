<template>
  <el-card shadow="never">
    <template #header>
      <div class="card-header">
        <span>老人档案</span>
        <div>
          <el-button size="small" type="primary" @click="load">刷新</el-button>
          <el-button size="small" type="success" @click="openAdd">新增档案</el-button>
        </div>
      </div>
    </template>
    <el-table :data="residents" size="small">
      <el-table-column prop="id" label="ID" width="60" />
      <el-table-column prop="name" label="姓名" width="120" />
      <el-table-column prop="gender" label="性别" width="70" />
      <el-table-column prop="age" label="年龄" width="70" />
      <el-table-column prop="guardian_name" label="家属/护工" width="120" />
      <el-table-column prop="guardian_phone" label="联系电话" width="140" />
      <el-table-column prop="address" label="住址" />
      <el-table-column prop="medical_history" label="病史" />
    </el-table>

    <el-dialog v-model="dialogVisible" title="老人档案" width="520px">
      <el-form :model="form" label-width="100px">
        <el-form-item label="姓名"><el-input v-model="form.name" /></el-form-item>
        <el-form-item label="性别"><el-input v-model="form.gender" /></el-form-item>
        <el-form-item label="年龄"><el-input v-model="form.age" type="number" /></el-form-item>
        <el-form-item label="家属姓名"><el-input v-model="form.guardian_name" /></el-form-item>
        <el-form-item label="家属电话"><el-input v-model="form.guardian_phone" /></el-form-item>
        <el-form-item label="住址"><el-input v-model="form.address" /></el-form-item>
        <el-form-item label="病史"><el-input v-model="form.medical_history" type="textarea" /></el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" @click="onAdd">确定</el-button>
      </template>
    </el-dialog>
  </el-card>
</template>

<script setup>
import { onMounted, ref, reactive } from 'vue'
import { ElMessage } from 'element-plus'
import api from '../api'

const residents = ref([])
const dialogVisible = ref(false)
const form = reactive({ name: '', gender: '男', age: null, guardian_name: '', guardian_phone: '', address: '', medical_history: '' })

async function load() {
  residents.value = await api.residents()
}
function openAdd() {
  Object.assign(form, { name: '', gender: '男', age: null, guardian_name: '', guardian_phone: '', address: '', medical_history: '' })
  dialogVisible.value = true
}
async function onAdd() {
  await api.addResident(form)
  ElMessage.success('档案已保存')
  dialogVisible.value = false
  load()
}
onMounted(load)
</script>

<style scoped>
.card-header { display: flex; justify-content: space-between; align-items: center; }
</style>